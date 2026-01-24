#!/usr/bin/env python3
"""
bQ Vector Analysis Script

Analyzes b_Q alignment patterns in trained models:
1. Estimates mean b_Q vectors over multiple batches
2. Analyzes token alignment (which tokens align with b_Q direction)
3. Analyzes positional alignment (positional patterns in b_Q direction)

Generates:
- Text results file
- b_Q mean norm plot
- Positional alignment heatmap
- Layer-wise positional alignment plots
"""

import os
import sys
import math
import pickle
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tiktoken
except ImportError:
    print("FATAL ERROR: tiktoken not found. Please run 'pip install tiktoken'")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecd_symbreak.config import GPTConfig


def log_output(message, f):
    """Prints to console and writes to the log file."""
    print(message)
    if f:
        f.write(message + "\n")


@dataclass
class GPTConfigLocal:
    context_length: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class DisorderedCausalSelfAttention(nn.Module):
    def __init__(self, config, mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mode="per_batch", use_k_bias=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        def to_vec(x, default, linspace_default=None):
            if x is None:
                x = linspace_default if linspace_default is not None else default
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 0:
                x = x.repeat(self.d)
            if x.shape != (self.d,):
                raise ValueError(f"Expected shape {(self.d,)} but got {x.shape}")
            return x.view(self.d)

        mean_Q = to_vec(mean_Q, 0.5)
        mean_K = to_vec(mean_K, 0.3)
        std_Q = to_vec(std_Q, None, linspace_default=torch.linspace(0.05, 0.15, self.d))
        std_K = to_vec(std_K, None, linspace_default=torch.linspace(0.12, 0.08, self.d))

        self.register_buffer("mean_Q", mean_Q)
        self.register_buffer("mean_K", mean_K)
        self.register_buffer("std_Q", std_Q.abs())
        self.register_buffer("std_K", std_K.abs())
        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))
        self.mode = mode
        self.use_k_bias = use_k_bias

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    @torch.no_grad()
    def _resample_biases(self, device):
        mQ, sQ = self.mean_Q.to(device), self.std_Q.to(device)
        mK, sK = self.mean_K.to(device), self.std_K.to(device)
        base_Q = mQ + sQ * torch.randn_like(mQ)
        base_K = mK + sK * torch.randn_like(mK)
        self.bQ.copy_(base_Q.unsqueeze(0).expand(self.n_head, -1).contiguous())
        self.bK.copy_(base_K.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        if self.mode == "per_batch":
            self._resample_biases(x.device)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1, 2)
        k = k.view(B, T, nh, d).transpose(1, 2)
        v = v.view(B, T, nh, d).transpose(1, 2)

        q = q + self.bQ.view(1, nh, 1, d)
        if self.use_k_bias:
            k = k + self.bK.view(1, nh, 1, d)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), self.bQ.detach()


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
        self.act = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, use_k_bias=True):
        super().__init__()
        self.sa = DisorderedCausalSelfAttention(config, mode="per_batch", use_k_bias=use_k_bias)
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        sa_out, bQ = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.mlp(self.ln2(x))
        return x, bQ


class GPT(nn.Module):
    def __init__(self, config, use_k_bias=True):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.context_length, config.n_embd),
            h=nn.ModuleList([Block(config, use_k_bias=use_k_bias) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
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

        bQs_list = []
        for block in self.transformer.h:
            x, bQ = block(x)
            bQs_list.append(bQ)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, bQs_list


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


def load_model_and_config(ckpt_path, device, f):
    log_output("\nLoading checkpoint...", f)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config_dict = ckpt['config']

    MODEL_PRESETS = {
        '124m': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
        '355m': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
        '774m': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
        '1.3b': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
    }

    if isinstance(config_dict, dict):
        if 'model' in config_dict and config_dict['model'] in MODEL_PRESETS:
            preset = MODEL_PRESETS[config_dict['model']]
            config_obj = GPTConfigLocal(
                context_length=config_dict.get('context_length', 512),
                vocab_size=config_dict.get('vocab_size', 50304),
                n_layer=preset['n_layer'],
                n_head=preset['n_head'],
                n_embd=preset['n_embd']
            )
        else:
            config_obj = GPTConfigLocal(
                context_length=config_dict.get('context_length', 512),
                vocab_size=config_dict.get('vocab_size', 50304),
                n_layer=config_dict.get('n_layer', 12),
                n_head=config_dict.get('n_head', 12),
                n_embd=config_dict.get('n_embd', 768)
            )
    else:
        config_obj = config_dict

    log_output(f"Model config: {config_obj.n_layer} layers, {config_obj.n_head} heads, "
               f"{config_obj.n_embd} embed dim.", f)
    log_output(f"Loaded config: vocab_size={config_obj.vocab_size}, "
               f"block_size={config_obj.context_length}", f)

    log_output("Loading model...", f)
    model = GPT(config_obj)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        raise KeyError(f"Checkpoint has neither 'model' nor 'model_state_dict'. "
                       f"Keys: {list(ckpt.keys())}")

    unfused_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unfused_state_dict, strict=False)
    model.to(device)
    model.eval()
    log_output("Model loaded.", f)

    return model, config_obj


@torch.no_grad()
def estimate_bQ_mean(model, data, num_batches, block_size, batch_size, device, f, output_prefix):
    log_output(f"\n--- Step 1: Estimating b_Q_mean over {num_batches} batches ---", f)
    bQ_sums = None
    bQ_counts = 0

    n_layer = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    for i in range(num_batches):
        if (i+1) % 10 == 0:
            log_output(f"  Batch {i+1}/{num_batches}", f)

        x, y = get_batch(data, block_size, batch_size, device)
        _, _, bQs_list = model(x, y)

        bQs_tensor = torch.stack(bQs_list).cpu()
        bQ_vector_per_layer = bQs_tensor[:, 0, :]  # Shape: [n_layer, d_head]

        if bQ_sums is None:
            bQ_sums = bQ_vector_per_layer
        else:
            bQ_sums += bQ_vector_per_layer

        bQ_counts += 1

    bQ_mean = bQ_sums / bQ_counts
    log_output("Estimation complete.", f)

    # Plot norm of mean b_Q per layer
    bQ_mean_norm = torch.norm(bQ_mean, p=2, dim=1).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_layer), bQ_mean_norm)
    plt.xlabel('Layer Index')
    plt.ylabel('Norm of mean b_Q')
    plt.title('Norm of Estimated Mean b_Q per Layer (Shared Vector)')
    plt.xticks(range(n_layer))
    plt.grid(True, axis='y', linestyle=':')

    save_path = f"{output_prefix}_bQ_mean_norm.png"
    plt.savefig(save_path)
    log_output(f"Saved b_Q mean norm plot to: {save_path}", f)
    plt.close()

    return bQ_mean.to(device)


@torch.no_grad()
def analyze_token_alignment(model, bQ_mean, itos, top_k, f, device):
    log_output(f"\n\n--- Step 2: Token Alignment Analysis ---", f)
    log_output("=" * 40, f)

    try:
        token_embeddings = model.transformer.wte.weight.to(device)
    except Exception as e:
        log_output(f"Error accessing token embeddings: {e}", f)
        return None

    n_layer, d_head = bQ_mean.shape
    n_head = model.config.n_head
    n_embd = token_embeddings.shape[1]
    vocab_size = token_embeddings.shape[0]

    results = {}

    for L in range(n_layer):
        log_output(f"\n{'='*50}\nLAYER {L}\n{'='*50}\n", f)
        results[L] = {}

        bQ_vec = bQ_mean[L, :]
        log_output(f"  (Comparing all heads against the SAME shared bQ_vec)", f)

        try:
            w_qkv = model.transformer.h[L].sa.c_attn.weight.to(device)
        except Exception as e:
            log_output(f"Skipping Layer {L}: Could not find sa.c_attn.weight. ({e})", f)
            continue

        w_q, w_k, w_v = w_qkv.split(n_embd, dim=0)
        all_keys = token_embeddings @ w_k.T
        all_keys_headed = all_keys.view(vocab_size, n_head, d_head)

        for H in range(n_head):
            all_keys_head = all_keys_headed[:, H, :]
            sims = F.cosine_similarity(all_keys_head, bQ_vec.unsqueeze(0), dim=1)

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


@torch.no_grad()
def analyze_positional_alignment(model, bQ_mean, block_size, f, output_prefix, device):
    log_output(f"\n\n--- Step 3: Positional Alignment Analysis ---", f)
    log_output("=" * 40, f)

    try:
        pos_embeddings = model.transformer.wpe.weight.to(device)
    except Exception as e:
        log_output(f"Error accessing positional embeddings: {e}", f)
        return None

    if pos_embeddings.shape[0] < block_size:
        log_output(f"  Warning: Positional embedding table ({pos_embeddings.shape[0]}) < "
                   f"block_size ({block_size}).", f)
        block_size = pos_embeddings.shape[0]
    elif pos_embeddings.shape[0] > block_size:
        log_output(f"  Positional embedding table ({pos_embeddings.shape[0]}) > "
                   f"block_size ({block_size}). Truncating.", f)
        pos_embeddings = pos_embeddings[:block_size, :]

    n_layer, d_head = bQ_mean.shape
    n_head = model.config.n_head
    n_embd = pos_embeddings.shape[1]

    sim_matrix = torch.zeros(n_layer, n_head, block_size, device='cpu')

    log_output("Processing layers...", f)
    for L in range(n_layer):
        bQ_vec = bQ_mean[L, :]

        try:
            w_qkv = model.transformer.h[L].sa.c_attn.weight.to(device)
        except Exception as e:
            log_output(f"Skipping Layer {L}: Could not find sa.c_attn.weight. ({e})", f)
            continue

        w_q, w_k, w_v = w_qkv.split(n_embd, dim=0)
        all_pos_keys = pos_embeddings @ w_k.T
        all_pos_keys_headed = all_pos_keys.view(block_size, n_head, d_head)

        for H in range(n_head):
            all_pos_keys_head = all_pos_keys_headed[:, H, :]
            sims = F.cosine_similarity(all_pos_keys_head, bQ_vec.unsqueeze(0), dim=1)
            sim_matrix[L, H, :] = sims.cpu()

    log_output("Plotting results...", f)

    # Plot 1: Heatmap
    sim_avg_heads = sim_matrix.mean(dim=1).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(sim_avg_heads, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5,
               interpolation='none')
    plt.colorbar(label='Avg. Cosine Similarity')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Layer Index')
    plt.title('Positional Alignment (Averaged Over Heads, vs. Shared Layer-wise bQ)')

    save_path_heatmap = f"{output_prefix}_positional_alignment_heatmap.png"
    plt.savefig(save_path_heatmap)
    log_output(f"Saved positional alignment heatmap to: {save_path_heatmap}", f)
    plt.close()

    # Plot 2: Line plots
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
    save_path_layers = f"{output_prefix}_positional_alignment_layers.png"
    plt.savefig(save_path_layers)
    log_output(f"Saved layer-wise positional plots to: {save_path_layers}", f)
    plt.close()

    return sim_matrix


def main():
    parser = argparse.ArgumentParser(description='Analyze b_Q alignment in trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output files')
    parser.add_argument('--data_path', type=str,
                        default="./data/finewebedu10B/finewebedu_train_000099.bin",
                        help='Path to training data')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches for b_Q mean estimation')
    parser.add_argument('--top_k', type=int, default=15,
                        help='Number of top/bottom tokens to show')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for analysis')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESULTS_FILENAME = f"{args.output_prefix}_results.txt"

    with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
        log_output(f"Starting b_Q analysis script...", f)
        log_output(f"Log file will be saved to: {RESULTS_FILENAME}", f)
        log_output(f"PyTorch version: {torch.__version__}", f)
        log_output(f"Using device: {DEVICE}", f)
        log_output(f"Checkpoint path: {args.checkpoint}", f)
        log_output(f"Data path: {args.data_path}", f)

        # Load Data
        log_output(f"\nLoading data from {args.data_path}...", f)
        try:
            train_data = np.memmap(args.data_path, dtype=np.uint16, mode='r')
            log_output(f"Data loaded. Total tokens: {len(train_data):,}", f)
        except FileNotFoundError:
            log_output(f"FATAL ERROR: Data file not found at {args.data_path}", f)
            return
        except Exception as e:
            log_output(f"FATAL ERROR: Could not load data. {e}", f)
            return

        # Load Model
        try:
            model, config_obj = load_model_and_config(args.checkpoint, DEVICE, f)
        except FileNotFoundError:
            log_output(f"FATAL ERROR: Checkpoint file not found at {args.checkpoint}", f)
            return
        except Exception as e:
            log_output(f"FATAL ERROR: Failed to load model. {e}", f)
            return

        block_size = config_obj.context_length
        vocab_size = config_obj.vocab_size

        # Load Vocab
        meta_path = os.path.join(os.path.dirname(args.data_path), "meta.pkl")
        itos = load_vocab(meta_path, vocab_size, f)

        # Run Analysis
        bQ_mean = estimate_bQ_mean(
            model, train_data, args.num_batches, block_size,
            args.batch_size, DEVICE, f, args.output_prefix
        )

        analyze_token_alignment(model, bQ_mean, itos, args.top_k, f, DEVICE)

        analyze_positional_alignment(
            model, bQ_mean, block_size, f, args.output_prefix, DEVICE
        )

        log_output("\n\n--- Analysis Complete ---", f)
        log_output(f"Text results saved to {RESULTS_FILENAME}", f)
        log_output("Plots saved:", f)
        log_output(f"  - {args.output_prefix}_bQ_mean_norm.png", f)
        log_output(f"  - {args.output_prefix}_positional_alignment_heatmap.png", f)
        log_output(f"  - {args.output_prefix}_positional_alignment_layers.png", f)


if __name__ == "__main__":
    main()
