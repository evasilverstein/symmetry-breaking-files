#!/usr/bin/env python
"""
bQ Vector Analysis Script (Batch version - accepts command-line args)

Modified to accept checkpoint path and output prefix as arguments.
Analyzes b_Q alignment patterns in trained models.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from types import SimpleNamespace
from dataclasses import dataclass
import math
import torch.nn as nn

# Use 'Agg' backend for non-interactive plotting (e.g., on servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- NEW V9 DEPENDENCY ---
try:
    import tiktoken
except ImportError:
    print("FATAL ERROR: tiktoken not found. Please run 'pip install tiktoken'")
    exit(1)
# --- END V9 DEPENDENCY ---

# --- NEW V10 HELPER ---
def log_output(message, f):
    """Prints to console and writes to the log file."""
    print(message)
    if f:
        f.write(message + "\n")
# --- END V10 HELPER ---

# =============================================================================
# --- MODEL DEFINITION (MODIFIED FOR V12) ---
# --- Matches 'train_scale_disorder_ecd (1).py' ---
# =============================================================================

@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768

class DisorderedCausalSelfAttention(nn.Module):
    def __init__(self, config, mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mode="per_batch", use_k_bias=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C); self.c_proj.NANOGPT_SCALE_INIT = 1

        # --- V12 MODIFICATION ---
        # Reverted to the 1D `to_vec` helper for ALL buffers,
        # matching the old training script.
        def to_vec(x, default, linspace_default=None):
            if x is None: 
                x = linspace_default if linspace_default is not None else default
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 0: x = x.repeat(self.d)
            if x.shape != (self.d,): 
                raise ValueError(f"1D Buffer init error: expected shape {(self.d,)} but got {x.shape}")
            return x.view(self.d)
        
        mean_Q = to_vec(mean_Q, 0.5)
        mean_K = to_vec(mean_K, 0.3)
        std_Q  = to_vec(std_Q,  None, linspace_default=torch.linspace(0.05, 0.15, self.d))
        std_K  = to_vec(std_K,  None, linspace_default=torch.linspace(0.12, 0.08,  self.d))
        # --- END V12 MODIFICATION ---

        self.register_buffer("mean_Q", mean_Q) # [d]
        self.register_buffer("mean_K", mean_K) # [d]
        self.register_buffer("std_Q",  std_Q.abs()) # [d]
        self.register_buffer("std_K",  std_K.abs()) # [d]
        
        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))
        self.mode = mode
        self.use_k_bias = use_k_bias

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1,1,config.context_length,config.context_length))

    @torch.no_grad()
    def _resample_biases(self, device):
        # --- V12 MODIFICATION ---
        # Reverted to old sampling logic
        mQ, sQ = self.mean_Q.to(device), self.std_Q.to(device) # 1D
        mK, sK = self.mean_K.to(device), self.std_K.to(device) # 1D
        base_Q = mQ + sQ * torch.randn_like(mQ) # 1D
        base_K = mK + sK * torch.randn_like(mK) # 1D
        
        # Expand the single 1D vector across all heads
        self.bQ.copy_(base_Q.unsqueeze(0).expand(self.n_head, -1).contiguous())
        self.bK.copy_(base_K.unsqueeze(0).expand(self.n_head, -1).contiguous())
        # --- END V12 MODIFICATION ---

    def forward(self, x):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        if self.mode == "per_batch":
            self._resample_biases(x.device)

        qkv = self.c_attn(x); q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1,2)
        k = k.view(B, T, nh, d).transpose(1,2)
        v = v.view(B, T, nh, d).transpose(1,2)

        q = q + self.bQ.view(1, nh, 1, d)
        if self.use_k_bias:
            k = k + self.bK.view(1, nh, 1, d)

        scores = (q @ k.transpose(-2,-1)) / math.sqrt(d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(scores, dim=-1)
        
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        # Return the bQ buffer for this layer
        return self.c_proj(y), self.bQ.detach()

# --- (Other model classes are identical to v11) ---

class RandomPReLU1d(nn.Module):
    def __init__(self, features, init_slope=0.2, slope_std=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features) * slope_std + init_slope)
    def forward(self, x):
        x2 = x.transpose(1, 2)
        y2 = F.prelu(x2, self.weight)
        return y2.transpose(1, 2)

class AsymmetricMLPPreLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act  = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd); self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config, use_k_bias=True):
        super().__init__()
        self.sa  = DisorderedCausalSelfAttention(config, mode="per_batch", use_k_bias=use_k_bias)
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
    def forward(self, x):
        # Modified to return bQ
        sa_out, bQ = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.mlp(self.ln2(x))
        return x, bQ

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, use_k_bias=True):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.context_length, config.n_embd),
            h    = nn.ModuleList([Block(config, use_k_bias=use_k_bias) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** (-0.5)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_length
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        
        # Modified to return bQs
        bQs_list = []
        for block in self.transformer.h:
            x, bQ = block(x)
            bQs_list.append(bQ)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        # Return list of [n_head, d_head] tensors
        return logits, loss, bQs_list

# =============================================================================
# --- END OF MODEL DEFINITION ---
# =============================================================================


# --- Parameters & Paths ---
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze b_Q alignment in trained models')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files')
parser.add_argument('--data_path', type=str, default="/workspace/modded-nanogpt/data/finewebedu10B/finewebedu_train_000099.bin", help='Path to training data')
args = parser.parse_args()

CKPT_PATH = args.checkpoint
BIN_PATH = args.data_path
META_PATH = os.path.join(os.path.dirname(BIN_PATH), "meta.pkl")
RESULTS_FILENAME = f"{args.output_prefix}_results.txt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Analysis Params ---
NUM_BATCHES_FOR_BQ_MEAN = 100
TOP_K_TOKENS = 15
ANALYSIS_BATCH_SIZE = 32


# --- Data Loading Utility ---

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Vocab Loading Utility (v9) ---
def load_vocab(meta_path, vocab_size, f):
    log_output("\nLoading vocab...", f)
    itos = {}

    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f_meta:
                meta = pickle.load(f_meta)
            if 'itos' in meta:
                log_output("meta.pkl found and loaded.", f)
                itos = meta['itos']
                if len(itos) < vocab_size:
                    log_output(f"Padding meta.pkl vocab ({len(itos)}) to model size ({vocab_size})...", f)
                    for i in range(len(itos), vocab_size):
                        itos[i] = f"[PAD:{i}]"
                return itos
        except Exception as e:
            log_output(f"Warning: meta.pkl found but failed to load: {e}.", f)

    log_output("meta.pkl not found or invalid. Attempting to load 'gpt2' tokenizer from tiktoken...", f)
    try:
        enc = tiktoken.get_encoding("gpt2")
        n_vocab_tiktoken = enc.n_vocab
        log_output(f"Successfully loaded 'gpt2' tokenizer (vocab size: {n_vocab_tiktoken})", f)
        
        for i in range(n_vocab_tiktoken):
            try:
                itos[i] = repr(enc.decode([i]))
            except Exception:
                itos[i] = f"[DECODE_ERR:{i}]"

        log_output(f"Built map for {len(itos)} tokens from 'gpt2' tokenizer.", f)
        
        if vocab_size > n_vocab_tiktoken:
            log_output(f"Model vocab size ({vocab_size}) > tokenizer ({n_vocab_tiktoken}). Adding padding tokens...", f)
            for i in range(n_vocab_tiktoken, vocab_size):
                itos[i] = f"[PAD:{i}]"
        
        log_output(f"Final itos map size: {len(itos)}", f)
        return itos
        
    except Exception as e:
        log_output(f"--- WARNING: tiktoken 'gpt2' failed to load: {e}", f)
        log_output("--- Reverting to fallback: [ID:1234]...", f)
        return {i: f"[ID:{i}]" for i in range(vocab_size)}


def decode(ids_list, itos):
    if isinstance(ids_list, torch.Tensor):
        ids_list = ids_list.tolist()
    return [itos.get(i, f"[ID:{i}]") for i in ids_list]


# --- Model Loading ---
def load_model_and_config(ckpt_path, device, f):
    log_output("\nLoading checkpoint...", f)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    config_dict = ckpt['config']

    # Convert dict to GPTConfig object
    if isinstance(config_dict, dict):
        config_obj = GPTConfig(
            context_length=config_dict.get('context_length', 512),
            vocab_size=config_dict.get('vocab_size', 50304),
            n_layer=config_dict.get('n_layer', 12),
            n_head=config_dict.get('n_head', 12),
            n_embd=config_dict.get('n_embd', 768)
        )
    else:
        config_obj = config_dict

    block_size = config_obj.context_length
    n_layer = config_obj.n_layer
    n_head = config_obj.n_head
    n_embd = config_obj.n_embd
    vocab_size = config_obj.vocab_size
    d_head = n_embd // n_head

    log_output(f"Model config: {n_layer} layers, {n_head} heads, {n_embd} embed dim.", f)
    log_output(f"Loaded config: vocab_size={vocab_size}, block_size={block_size}", f)

    log_output("Loading REAL model (single mean bQ version)...", f)
    model = GPT(config_obj)
    log_output("Note: Assumes model was trained with use_k_bias=True (the default).", f)

    state_dict = ckpt['model']
    unfused_state_dict = {k.replace('_orig_mod.',''): v for k, v in state_dict.items()}

    model.load_state_dict(unfused_state_dict, strict=False)
    model.to(device)
    model.eval()
    log_output("Actual model loaded.", f)

    return model, config_obj

# --- Analysis Function 1: Estimate b_Q Mean (V12 MODIFIED) ---

@torch.no_grad()
def estimate_bQ_mean(model, data, num_batches, block_size, batch_size, device, f):
    log_output(f"\n--- Step 1: Estimating b_Q_mean over {num_batches} batches (Single-Mean-Vector-Per-Layer) ---", f)
    bQ_sums = None
    bQ_counts = 0
    
    n_layer = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    
    for i in range(num_batches):
        if (i+1) % 10 == 0:
            log_output(f"  Batch {i+1}/{num_batches}", f)
        
        x, y = get_batch(data, block_size, batch_size, device)
        _, _, bQs_list = model(x, y) 
        
        # bQs_tensor has shape [n_layer, n_head, d_head]
        bQs_tensor = torch.stack(bQs_list).cpu() 
        
        # --- V12 MODIFICATION ---
        # In this model, all heads in a layer share the *same* bQ.
        # So we only need to average the vector from the first head (or any head).
        # We store a 2D tensor of shape [n_layer, d_head]
        bQ_vector_per_layer = bQs_tensor[:, 0, :] # Shape: [n_layer, d_head]
        
        if bQ_sums is None:
            bQ_sums = bQ_vector_per_layer
        else:
            bQ_sums += bQ_vector_per_layer
        # --- END V12 MODIFICATION ---
        
        bQ_counts += 1
        
    bQ_mean = bQ_sums / bQ_counts # Final shape: [n_layer, d_head]
    log_output("Estimation complete.", f)
    
    # --- V12 MODIFICATION: New Plot ---
    # We now have one vector per layer, so a heatmap is not useful.
    # We will plot a bar chart of the norm of the mean b_Q for each layer.
    bQ_mean_norm = torch.norm(bQ_mean, p=2, dim=1).cpu().numpy() # dim=1 is d_head
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_layer), bQ_mean_norm)
    plt.xlabel('Layer Index')
    plt.ylabel('Norm of mean b_Q')
    plt.title('Norm of Estimated Mean b_Q per Layer (Shared Vector)')
    plt.xticks(range(n_layer))
    plt.grid(True, axis='y', linestyle=':')
    
    save_path = f"{args.output_prefix}_bQ_mean_norm.png"
    plt.savefig(save_path)
    log_output(f"Saved b_Q mean norm plot to: {save_path}", f)
    plt.close()
    # --- END V12 MODIFICATION ---
    
    return bQ_mean.to(device) # Shape: [n_layer, d_head]


# --- Analysis Function 2: Token Alignment (V12 MODIFIED) ---

@torch.no_grad()
def analyze_token_alignment(model, bQ_mean, itos, top_k, f):
    log_output(f"\n\n--- Step 2: Token Alignment Analysis (Shared-Mean-Vector-Per-Layer) ---", f)
    log_output("=" * 40, f)
    
    try:
        token_embeddings = model.transformer.wte.weight.to(DEVICE)
    except Exception as e:
        log_output(f"Error accessing token embeddings (model.transformer.wte.weight): {e}", f)
        return None
        
    # --- V12 MODIFICATION ---
    # bQ_mean is now 2D: [n_layer, d_head]
    n_layer, d_head = bQ_mean.shape
    n_head = model.config.n_head
    # --- END V12 MODIFICATION ---
    
    n_embd = token_embeddings.shape[1]
    vocab_size = token_embeddings.shape[0]
    
    results = {}
    
    for L in range(n_layer):
        log_output(f"\n{'='*50}\nLAYER {L}\n{'='*50}\n", f)
        results[L] = {}
        
        # --- V12 MODIFICATION ---
        # Get the single, shared bQ vector for this layer
        bQ_vec = bQ_mean[L, :] # Shape: [d_head]
        log_output(f"  (Comparing all heads in this layer against the SAME shared bQ_vec)", f)
        # --- END V12 MODIFICATION ---
        
        try:
            w_qkv = model.transformer.h[L].sa.c_attn.weight.to(DEVICE)
        except Exception as e:
            log_output(f"Skipping Layer {L}: Could not find sa.c_attn.weight. ({e})", f)
            continue
            
        w_q, w_k, w_v = w_qkv.split(n_embd, dim=0)
        all_keys = token_embeddings @ w_k.T
        all_keys_headed = all_keys.view(vocab_size, n_head, d_head)
        
        for H in range(n_head):
            all_keys_head = all_keys_headed[:, H, :]
            
            # --- V12 MODIFICATION ---
            # `bQ_vec` is now the same for all heads in this loop
            sims = F.cosine_similarity(all_keys_head, bQ_vec.unsqueeze(0), dim=1)
            # --- END V12 MODIFICATION ---
            
            top_vals, top_indices = torch.topk(sims, top_k)
            bot_vals, bot_indices = torch.topk(sims, top_k, largest=False)
            
            top_tokens = list(zip(decode(top_indices, itos), top_vals.cpu().tolist()))
            bot_tokens = list(zip(decode(bot_indices, itos), bot_vals.cpu().tolist()))
            
            results[L][H] = {'top': top_tokens, 'bottom': bot_tokens}
            
            log_output(f"\n  --- Head {H:2d} ---", f)
            log_output("    [Top Align]", f)
            for i, (token, val) in enumerate(top_tokens):
                log_output(f"        {i+1:2d}. {token:<20} ({val:+.4f})", f)
                
            log_output("\n    [Bottom Align]", f)
            for i, (token, val) in enumerate(bot_tokens):
                log_output(f"        {i+1:2d}. {token:<20} ({val:+.4f})", f)
            
    return results

# --- Analysis Function 3: Positional Alignment (V12 MODIFIED) ---

@torch.no_grad()
def analyze_positional_alignment(model, bQ_mean, block_size, f):
    log_output(f"\n\n--- Step 3: Positional Alignment Analysis (Shared-Mean-Vector-Per-Layer) ---", f)
    log_output("=" * 40, f)
    
    try:
        pos_embeddings = model.transformer.wpe.weight.to(DEVICE)
    except Exception as e:
        log_output(f"Error accessing positional embeddings (model.transformer.wpe.weight): {e}", f)
        return None

    if pos_embeddings.shape[0] < block_size:
        log_output(f"  Warning: Positional embedding table ({pos_embeddings.shape[0]}) < block_size ({block_size}).", f)
        block_size = pos_embeddings.shape[0]
        log_output(f"  Analyzing first {block_size} positions only.", f)
    elif pos_embeddings.shape[0] > block_size:
        log_output(f"  Positional embedding table ({pos_embeddings.shape[0]}) > block_size ({block_size}). Truncating.", f)
        pos_embeddings = pos_embeddings[:block_size, :]
        
    # --- V12 MODIFICATION ---
    n_layer, d_head = bQ_mean.shape
    n_head = model.config.n_head
    # --- END V12 MODIFICATION ---
    n_embd = pos_embeddings.shape[1]
    
    sim_matrix = torch.zeros(n_layer, n_head, block_size, device='cpu')
    
    log_output("Processing layers...", f)
    for L in range(n_layer):
        # --- V12 MODIFICATION ---
        # Get the single, shared bQ vector for this layer
        bQ_vec = bQ_mean[L, :] # Shape: [d_head]
        # --- END V12 MODIFICATION ---
        
        try:
            w_qkv = model.transformer.h[L].sa.c_attn.weight.to(DEVICE)
        except Exception as e:
            log_output(f"Skipping Layer {L}: Could not find sa.c_attn.weight. ({e})", f)
            continue
            
        w_q, w_k, w_v = w_qkv.split(n_embd, dim=0)
        all_pos_keys = pos_embeddings @ w_k.T
        all_pos_keys_headed = all_pos_keys.view(block_size, n_head, d_head)
        
        for H in range(n_head):
            all_pos_keys_head = all_pos_keys_headed[:, H, :]
            
            # --- V12 MODIFICATION ---
            # `bQ_vec` is now the same for all heads in this loop
            sims = F.cosine_similarity(all_pos_keys_head, bQ_vec.unsqueeze(0), dim=1)
            # --- END V12 MODIFICATION ---
            
            sim_matrix[L, H, :] = sims.cpu()
            
    log_output("Plotting results...", f)
    
    # Plot 1: Heatmap (This plot is still useful!)
    # It shows how different heads (y-axis) in different layers (x-axis)
    # align with the *shared* bQ vector for that layer.
    sim_avg_heads = sim_matrix.mean(dim=1).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(sim_avg_heads, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5, interpolation='none')
    plt.colorbar(label='Avg. Cosine Similarity')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Layer Index')
    plt.title('Positional Alignment (Averaged Over Heads, vs. Shared Layer-wise bQ)')
    
    save_path_heatmap = f"{args.output_prefix}_positional_alignment_heatmap.png"
    plt.savefig(save_path_heatmap)
    log_output(f"Saved positional alignment heatmap to: {save_path_heatmap}", f)
    plt.close()
    
    # Plot 2: Line plots (Also still useful)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sim_matrix[0, :, :].T.numpy())
    plt.title(f'Layer 0: Positional Alignment (All {n_head} Heads vs. Shared bQ)')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, linestyle=':')
    plt.ylim(-1, 1)
    
    plt.subplot(1, 2, 2)
    plt.plot(sim_matrix[n_layer-1, :, :].T.numpy())
    plt.title(f'Layer {n_layer-1}: Positional Alignment (All {n_head} Heads vs. Shared bQ)')
    plt.xlabel('Position in Sequence')
    plt.grid(True, linestyle=':')
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    save_path_layers = f"{args.output_prefix}_positional_alignment_layers.png"
    plt.savefig(save_path_layers)
    log_output(f"Saved layer-wise positional plots to: {save_path_layers}", f)
    plt.close()
    
    return sim_matrix

# --- Main Execution ---

def main():
    with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
        log_output(f"Starting b_Q analysis script (v12 - Single Mean)...", f)
        log_output(f"Log file will be saved to: {RESULTS_FILENAME}", f)
        log_output(f"PyTorch version: {torch.__version__}", f)
        log_output(f"tiktoken import successful: {('tiktoken' in globals())}", f)
        log_output(f"Using device: {DEVICE}", f)
        log_output(f"Checkpoint path: {CKPT_PATH}", f)
        log_output(f"Data path: {BIN_PATH}", f)

        # 1. Load Data
        log_output(f"\nLoading data from {BIN_PATH}...", f)
        try:
            train_data = np.memmap(BIN_PATH, dtype=np.uint16, mode='r')
            log_output(f"Data loaded. Total tokens: {len(train_data):,}", f)
        except FileNotFoundError:
            log_output(f"FATAL ERROR: Data file not found at {BIN_PATH}", f)
            log_output("Please check the `BIN_PATH` variable in the script.", f)
            return
        except Exception as e:
            log_output(f"FATAL ERROR: Could not load data. {e}", f)
            return
        
        # 2. Load Model
        try:
            model, config_obj = load_model_and_config(CKPT_PATH, DEVICE, f)
        except FileNotFoundError:
            log_output(f"FATAL ERROR: Checkpoint file not found at {CKPT_PATH}", f)
            log_output("Please check the `CKPT_PATH` variable in the script.", f)
            return
        except RuntimeError as e:
            log_output(f"FATAL ERROR: Failed to load model. {e}", f)
            log_output("\n--- COMMON CAUSE ---", f)
            log_output("This *likely* means the checkpoint you are loading was trained with the", f)
            log_output("*per-head* bQ script, but you are running the *single-mean* analysis script.", f)
            log_output("Please use 'analyze_bQ_v11.py' for that model.", f)
            return
        except Exception as e:
            log_output(f"FATAL ERROR: Failed to load model. {e}", f)
            return
            
        block_size = config_obj.context_length
        vocab_size = config_obj.vocab_size
        
        # 3. Load Vocab
        itos = load_vocab(META_PATH, vocab_size, f)
        
        # 4. Run Analysis
        bQ_mean = estimate_bQ_mean(
            model, 
            train_data, 
            NUM_BATCHES_FOR_BQ_MEAN, 
            block_size, 
            ANALYSIS_BATCH_SIZE, 
            DEVICE,
            f=f
        )
        
        token_results = analyze_token_alignment(
            model, 
            bQ_mean, 
            itos, 
            TOP_K_TOKENS,
            f=f
        )
        
        pos_results = analyze_positional_alignment(
            model, 
            bQ_mean, 
            block_size,
            f=f
        )
        
        log_output("\n\n--- Analysis Complete ---", f)
        log_output(f"Text results were printed above and saved to {RESULTS_FILENAME}", f)
        log_output("Plots were saved to:", f)
        log_output(f"  - {args.output_prefix}_bQ_mean_norm.png", f)
        log_output(f"  - {args.output_prefix}_positional_alignment_heatmap.png", f)
        log_output(f"  - {args.output_prefix}_positional_alignment_layers.png", f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n--- SCRIPT CRASHED ---")
        print(f"Error: {e}")
        try:
            with open(RESULTS_FILENAME, 'a', encoding='utf-8') as f:
                f.write("\n\n--- SCRIPT CRASHED ---\n")
                f.write(f"Error: {e}\n")
                import traceback
                traceback.print_exc(file=f)
        except:
            pass


