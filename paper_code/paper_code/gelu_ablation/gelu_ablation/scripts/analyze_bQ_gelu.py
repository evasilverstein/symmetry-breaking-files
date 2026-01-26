#!/usr/bin/env python3
"""
bQ Vector Analysis Script for GELU Ablation Models

IMPORTANT: This script is SPECIFICALLY for GELU ablation models.
DO NOT USE for PReLU models - use the main workspace analysis scripts instead.

Analyzes b_Q alignment patterns in GELU-trained models.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from dataclasses import dataclass
import math
import torch.nn as nn

# Use 'Agg' backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tiktoken
except ImportError:
    print("FATAL ERROR: tiktoken not found. Please run 'pip install tiktoken'")
    sys.exit(1)

def log_output(message, f):
    """Prints to console and writes to the log file."""
    print(message)
    if f:
        f.write(message + "\n")

# =============================================================================
# MODEL DEFINITION - GELU VERSION
# =============================================================================

@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768


class DisorderedCausalSelfAttentionGELU(nn.Module):
    """Attention with bQ/bK biases - for GELU models."""

    def __init__(self, config, mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mode="per_batch", use_k_bias=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)

        def to_vec(x, default, linspace_default=None):
            if x is None:
                x = linspace_default if linspace_default is not None else default
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 0: x = x.repeat(self.d)
            if x.shape != (self.d,):
                raise ValueError(f"Buffer init error: expected {(self.d,)} got {x.shape}")
            return x.view(self.d)

        mean_Q = to_vec(mean_Q, 0.5)
        mean_K = to_vec(mean_K, 0.3)
        std_Q  = to_vec(std_Q,  None, linspace_default=torch.linspace(0.05, 0.15, self.d))
        std_K  = to_vec(std_K,  None, linspace_default=torch.linspace(0.12, 0.08, self.d))

        self.register_buffer("mean_Q", mean_Q)
        self.register_buffer("mean_K", mean_K)
        self.register_buffer("std_Q",  std_Q.abs())
        self.register_buffer("std_K",  std_K.abs())

        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))
        self.mode = mode
        self.use_k_bias = use_k_bias

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1,1,config.context_length,config.context_length))

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

        return self.c_proj(y), self.bQ.detach()


class GELUMLP(nn.Module):
    """Standard GELU MLP - the key difference from PReLU models."""
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.c_proj = nn.Linear(hidden, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, use_k_bias=True):
        super().__init__()
        self.sa  = DisorderedCausalSelfAttentionGELU(config, mode="per_batch", use_k_bias=use_k_bias)
        self.mlp = GELUMLP(config)  # GELU, not PReLU!
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
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


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def load_vocab(vocab_size, f):
    log_output("\nLoading vocab...", f)
    itos = {}

    try:
        enc = tiktoken.get_encoding("gpt2")
        n_vocab_tiktoken = enc.n_vocab
        log_output(f"Successfully loaded 'gpt2' tokenizer (vocab size: {n_vocab_tiktoken})", f)

        for i in range(n_vocab_tiktoken):
            try:
                itos[i] = repr(enc.decode([i]))
            except Exception:
                itos[i] = f"[DECODE_ERR:{i}]"

        if vocab_size > n_vocab_tiktoken:
            log_output(f"Model vocab size ({vocab_size}) > tokenizer ({n_vocab_tiktoken}). Adding padding...", f)
            for i in range(n_vocab_tiktoken, vocab_size):
                itos[i] = f"[PAD:{i}]"

        log_output(f"Final itos map size: {len(itos)}", f)
        return itos

    except Exception as e:
        log_output(f"WARNING: tiktoken failed: {e}", f)
        return {i: f"[ID:{i}]" for i in range(vocab_size)}


def main():
    parser = argparse.ArgumentParser(description='Analyze b_Q alignment in GELU models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files')
    parser.add_argument('--data_path', type=str,
                        default="/workspace/modded-nanogpt/data/finewebedu10B/finewebedu_train_000099.bin",
                        help='Path to training data')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_BATCHES_FOR_BQ_MEAN = 100
    TOP_K_TOKENS = 15
    ANALYSIS_BATCH_SIZE = 32

    # Create output file
    results_filename = f"{args.output_prefix}_results.txt"
    os.makedirs(os.path.dirname(results_filename) if os.path.dirname(results_filename) else '.', exist_ok=True)

    with open(results_filename, 'w') as f:
        log_output("Starting GELU b_Q analysis script...", f)
        log_output(f"Log file: {results_filename}", f)
        log_output(f"PyTorch version: {torch.__version__}", f)
        log_output(f"Using device: {DEVICE}", f)
        log_output(f"Checkpoint path: {args.checkpoint}", f)
        log_output(f"Data path: {args.data_path}", f)

        # Load data
        log_output(f"\nLoading data from {args.data_path}...", f)
        data = np.memmap(args.data_path, dtype=np.uint16, mode='r')
        log_output(f"Data loaded. Total tokens: {len(data):,}", f)

        # Load checkpoint
        log_output("\nLoading checkpoint...", f)
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        if 'config' in ckpt:
            cfg = ckpt['config']
            config = GPTConfig(
                context_length=getattr(cfg, 'context_length', 512),
                vocab_size=getattr(cfg, 'vocab_size', 50304),
                n_layer=getattr(cfg, 'n_layer', 12),
                n_head=getattr(cfg, 'n_head', 12),
                n_embd=getattr(cfg, 'n_embd', 768),
            )
        else:
            config = GPTConfig()

        log_output(f"Model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embed dim.", f)
        log_output(f"MLP type: GELU (ablation model)", f)

        # Build model
        model = GPT(config, use_k_bias=True)

        # Load state dict
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Clean state dict for loading
        cleaned = {}
        for k, v in state_dict.items():
            # Map key names
            k_clean = k
            if '.attn.' in k:
                k_clean = k.replace('.attn.', '.sa.')

            # Skip bias-related keys that don't match our model
            skip_keys = ['.bV', '.mean_V', '.std_V', '.rope', '.rotary']
            if any(sk in k_clean for sk in skip_keys):
                continue
            cleaned[k_clean] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        real_missing = [k for k in missing if not any(x in k for x in ['bQ', 'bK', 'bV'])]
        if real_missing:
            log_output(f"Warning: Missing keys: {real_missing[:5]}...", f)

        model = model.to(DEVICE)
        model.eval()

        # Load vocab
        itos = load_vocab(config.vocab_size, f)

        # Step 1: Estimate bQ mean
        log_output(f"\n--- Step 1: Estimating b_Q_mean over {NUM_BATCHES_FOR_BQ_MEAN} batches ---", f)

        bQ_accum = [torch.zeros(config.n_head, config.n_embd // config.n_head, device=DEVICE)
                    for _ in range(config.n_layer)]

        model.train()  # Enable bias resampling
        with torch.no_grad():
            for batch_idx in range(NUM_BATCHES_FOR_BQ_MEAN):
                x, _ = get_batch(data, config.context_length, ANALYSIS_BATCH_SIZE, DEVICE)
                _, _, bQs_list = model(x)
                for layer_idx, bQ_layer in enumerate(bQs_list):
                    bQ_accum[layer_idx] += bQ_layer
                if (batch_idx + 1) % 10 == 0:
                    log_output(f"  Batch {batch_idx + 1}/{NUM_BATCHES_FOR_BQ_MEAN}", f)

        bQ_means = [acc / NUM_BATCHES_FOR_BQ_MEAN for acc in bQ_accum]
        log_output("Estimation complete.", f)

        # Plot bQ mean norms
        plot_path = f"{args.output_prefix}_bQ_mean_norm.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        for layer_idx, bQ_mean in enumerate(bQ_means):
            norms = bQ_mean.norm(dim=1).cpu().numpy()
            ax.bar(np.arange(config.n_head) + layer_idx * 0.05, norms,
                   width=0.8/config.n_layer, label=f'L{layer_idx}', alpha=0.7)
        ax.set_xlabel('Head')
        ax.set_ylabel('bQ Mean Norm')
        ax.set_title('GELU Model: bQ Mean Norm by Layer and Head')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log_output(f"Saved bQ mean norm plot to: {plot_path}", f)

        # Step 2: Token Alignment Analysis
        log_output(f"\n--- Step 2: Token Alignment Analysis ---", f)

        model.eval()
        wte = model.transformer.wte.weight.detach()  # [vocab, n_embd]

        d_head = config.n_embd // config.n_head

        for layer_idx in range(config.n_layer):
            log_output(f"\n{'='*50}", f)
            log_output(f"LAYER {layer_idx}", f)
            log_output(f"{'='*50}", f)

            # Get WK for this layer
            c_attn_weight = model.transformer.h[layer_idx].sa.c_attn.weight.detach()
            WK = c_attn_weight[config.n_embd:2*config.n_embd, :]  # [n_embd, n_embd]

            # Compute WK * embedding for all tokens
            WK_embed = wte @ WK.T  # [vocab, n_embd]

            # Use mean bQ across all heads for this layer
            bQ_layer_mean = bQ_means[layer_idx].mean(dim=0)  # [d_head]

            log_output(f"\n  (Using layer-averaged bQ mean vector)", f)

            for head_idx in range(config.n_head):
                # Extract head-specific WK projection
                start_idx = head_idx * d_head
                end_idx = (head_idx + 1) * d_head
                WK_head = WK_embed[:, start_idx:end_idx]  # [vocab, d_head]

                # Compute alignment with bQ mean
                bQ_vec = bQ_layer_mean  # Use layer mean
                alignment = F.cosine_similarity(WK_head, bQ_vec.unsqueeze(0), dim=1)  # [vocab]

                # Get top and bottom aligned tokens
                top_vals, top_idxs = torch.topk(alignment, TOP_K_TOKENS)
                bot_vals, bot_idxs = torch.topk(-alignment, TOP_K_TOKENS)

                log_output(f"\n  --- Head {head_idx:2d} ---", f)
                log_output(f"    [Top Align]", f)
                for i in range(TOP_K_TOKENS):
                    tok_id = top_idxs[i].item()
                    tok_str = itos.get(tok_id, f"[ID:{tok_id}]")
                    log_output(f"         {i+1:2d}. {tok_str:20s} (+{top_vals[i].item():.4f})", f)

                log_output(f"    [Bot Align]", f)
                for i in range(TOP_K_TOKENS):
                    tok_id = bot_idxs[i].item()
                    tok_str = itos.get(tok_id, f"[ID:{tok_id}]")
                    log_output(f"         {i+1:2d}. {tok_str:20s} ({-bot_vals[i].item():.4f})", f)

        # Step 3: Positional alignment heatmap
        log_output(f"\n--- Step 3: Positional Alignment Heatmap ---", f)

        pos_embed = model.transformer.wpe.weight.detach()  # [context_length, n_embd]

        heatmap_data = np.zeros((config.n_layer, config.context_length))

        for layer_idx in range(config.n_layer):
            c_attn_weight = model.transformer.h[layer_idx].sa.c_attn.weight.detach()
            WK = c_attn_weight[config.n_embd:2*config.n_embd, :]
            WK_pos = pos_embed @ WK.T  # [context_length, n_embd]

            bQ_layer_mean = bQ_means[layer_idx].mean(dim=0)  # [d_head]

            # Average alignment across heads
            for head_idx in range(config.n_head):
                start_idx = head_idx * d_head
                end_idx = (head_idx + 1) * d_head
                WK_head = WK_pos[:, start_idx:end_idx]
                alignment = F.cosine_similarity(WK_head, bQ_layer_mean.unsqueeze(0), dim=1)
                heatmap_data[layer_idx] += alignment.cpu().numpy()

            heatmap_data[layer_idx] /= config.n_head

        heatmap_path = f"{args.output_prefix}_positional_alignment_heatmap.png"
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Layer')
        ax.set_title('GELU Model: Positional Alignment with bQ Mean')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        log_output(f"Saved positional alignment heatmap to: {heatmap_path}", f)

        # Plot per-layer positional alignment
        layers_path = f"{args.output_prefix}_positional_alignment_layers.png"
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        for layer_idx in range(min(config.n_layer, 12)):
            ax = axes[layer_idx]
            ax.plot(heatmap_data[layer_idx])
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Alignment')
            ax.set_ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.savefig(layers_path, dpi=150)
        plt.close()
        log_output(f"Saved per-layer positional alignment to: {layers_path}", f)

        log_output(f"\n{'='*50}", f)
        log_output("GELU bQ Analysis Complete!", f)
        log_output(f"{'='*50}", f)


if __name__ == "__main__":
    main()
