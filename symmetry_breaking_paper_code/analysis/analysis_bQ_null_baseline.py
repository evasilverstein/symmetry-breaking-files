#!/usr/bin/env python
"""
bQ Null Baseline Analysis Script

This script provides a null/control analysis by computing alignment
between W_K*x and RANDOM bQ vectors for a symmetric model (no learned bQ).

The purpose is to show what "random alignment" looks like, serving as
a baseline comparison for the learned alignments in disordered models.

For a symmetric model, any correlation between W_K*x and random bQ
should be purely statistical noise with ~zero mean.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from dataclasses import dataclass
import math
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tiktoken
except ImportError:
    print("FATAL ERROR: tiktoken not found. Please run 'pip install tiktoken'")
    exit(1)

def log_output(message, f):
    """Prints to console and writes to the log file."""
    print(message)
    if f:
        f.write(message + "\n")


# =============================================================================
# --- SYMMETRIC MODEL DEFINITION ---
# =============================================================================

@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention (no bQ/bK)."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1, 2)
        k = k.view(B, T, nh, d).transpose(1, 2)
        v = v.view(B, T, nh, d).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


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
    def __init__(self, config):
        super().__init__()
        self.sa = CausalSelfAttention(config)
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_length, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# =============================================================================
# --- ANALYSIS FUNCTIONS ---
# =============================================================================

def generate_random_bQ(n_layer, d_head, mean_Q=0.5, std_Q_min=0.05, std_Q_max=0.15):
    """Generate random bQ vectors matching the disordered model distribution."""
    bQ_random = torch.zeros(n_layer, d_head)
    std_Q = torch.linspace(std_Q_min, std_Q_max, d_head)

    for L in range(n_layer):
        bQ_random[L] = mean_Q + std_Q * torch.randn(d_head)

    return bQ_random


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
                    for i in range(len(itos), vocab_size):
                        itos[i] = f"[PAD:{i}]"
                return itos
        except Exception as e:
            log_output(f"Warning: meta.pkl found but failed to load: {e}.", f)

    log_output("Using tiktoken 'gpt2' tokenizer...", f)
    try:
        enc = tiktoken.get_encoding("gpt2")
        n_vocab_tiktoken = enc.n_vocab

        for i in range(n_vocab_tiktoken):
            try:
                itos[i] = repr(enc.decode([i]))
            except Exception:
                itos[i] = f"[DECODE_ERR:{i}]"

        if vocab_size > n_vocab_tiktoken:
            for i in range(n_vocab_tiktoken, vocab_size):
                itos[i] = f"[PAD:{i}]"

        return itos

    except Exception as e:
        log_output(f"Warning: tiktoken failed: {e}", f)
        return {i: f"[ID:{i}]" for i in range(vocab_size)}


def decode(ids_list, itos):
    if isinstance(ids_list, torch.Tensor):
        ids_list = ids_list.tolist()
    return [itos.get(i, f"[ID:{i}]") for i in ids_list]


def load_symmetric_model(ckpt_path, device, f):
    log_output(f"\nLoading symmetric checkpoint from {ckpt_path}...", f)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    config_dict = ckpt['config']

    # Build GPTConfig
    config_obj = GPTConfig(
        context_length=config_dict.get('context_length', 512),
        vocab_size=config_dict.get('vocab_size', 50304),
        n_layer=12,  # 124M model
        n_head=12,
        n_embd=768
    )

    log_output(f"Config: {config_obj.n_layer} layers, {config_obj.n_head} heads", f)
    log_output(f"use_disordered_attention: {config_dict.get('use_disordered_attention', 'N/A')}", f)

    model = GPT(config_obj)

    # Load state dict
    state_dict = ckpt['model_state_dict']
    # Remove _orig_mod prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    log_output("Symmetric model loaded.", f)
    return model, config_obj


@torch.no_grad()
def analyze_null_alignment(model, bQ_random, itos, top_k, device, f):
    """Analyze alignment between W_K*x and random bQ vectors."""
    log_output(f"\n\n--- NULL BASELINE: Token Alignment with Random bQ ---", f)
    log_output("=" * 50, f)
    log_output("This shows what alignment looks like with RANDOM bQ vectors", f)
    log_output("(no learned relationship between model weights and bQ)", f)
    log_output("=" * 50, f)

    token_embeddings = model.transformer.wte.weight.to(device)
    bQ_random = bQ_random.to(device)

    n_layer, d_head = bQ_random.shape
    n_head = model.config.n_head
    n_embd = token_embeddings.shape[1]
    vocab_size = token_embeddings.shape[0]

    results = {}

    for L in range(n_layer):
        log_output(f"\n{'='*50}\nLAYER {L}\n{'='*50}\n", f)
        results[L] = {}

        bQ_vec = bQ_random[L, :]  # Random bQ for this layer

        w_qkv = model.transformer.h[L].sa.c_attn.weight.to(device)
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
            log_output("    [Top Align (RANDOM)]", f)
            for i, (token, val) in enumerate(top_tokens):
                log_output(f"        {i+1:2d}. {token:<20} ({val:+.4f})", f)

            log_output("\n    [Bottom Align (RANDOM)]", f)
            for i, (token, val) in enumerate(bot_tokens):
                log_output(f"        {i+1:2d}. {token:<20} ({val:+.4f})", f)

    return results


@torch.no_grad()
def analyze_null_positional(model, bQ_random, block_size, device, output_prefix, f):
    """Analyze positional alignment with random bQ."""
    log_output(f"\n\n--- NULL BASELINE: Positional Alignment with Random bQ ---", f)
    log_output("=" * 50, f)

    pos_embeddings = model.transformer.wpe.weight.to(device)
    bQ_random = bQ_random.to(device)

    n_layer, d_head = bQ_random.shape
    n_head = model.config.n_head
    n_embd = pos_embeddings.shape[1]

    if pos_embeddings.shape[0] > block_size:
        pos_embeddings = pos_embeddings[:block_size, :]

    sim_matrix = torch.zeros(n_layer, n_head, block_size, device='cpu')

    for L in range(n_layer):
        bQ_vec = bQ_random[L, :]

        w_qkv = model.transformer.h[L].sa.c_attn.weight.to(device)
        w_q, w_k, w_v = w_qkv.split(n_embd, dim=0)

        all_pos_keys = pos_embeddings @ w_k.T
        all_pos_keys_headed = all_pos_keys.view(block_size, n_head, d_head)

        for H in range(n_head):
            all_pos_keys_head = all_pos_keys_headed[:, H, :]
            sims = F.cosine_similarity(all_pos_keys_head, bQ_vec.unsqueeze(0), dim=1)
            sim_matrix[L, H, :] = sims.cpu()

    # Plot heatmap
    sim_avg_heads = sim_matrix.mean(dim=1).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(sim_avg_heads, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5, interpolation='none')
    plt.colorbar(label='Avg. Cosine Similarity')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Layer Index')
    plt.title('NULL BASELINE: Positional Alignment with Random bQ\n(Symmetric Model - No Learned bQ)')

    save_path = f"{output_prefix}_positional_alignment_heatmap.png"
    plt.savefig(save_path)
    log_output(f"Saved: {save_path}", f)
    plt.close()

    # Plot layer comparison
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sim_matrix[0, :, :].T.numpy())
    plt.title(f'Layer 0: Positional Alignment (Random bQ)')
    plt.xlabel('Position')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, linestyle=':')
    plt.ylim(-1, 1)

    plt.subplot(1, 2, 2)
    plt.plot(sim_matrix[n_layer-1, :, :].T.numpy())
    plt.title(f'Layer {n_layer-1}: Positional Alignment (Random bQ)')
    plt.xlabel('Position')
    plt.grid(True, linestyle=':')
    plt.ylim(-1, 1)

    plt.tight_layout()
    save_path = f"{output_prefix}_positional_alignment_layers.png"
    plt.savefig(save_path)
    log_output(f"Saved: {save_path}", f)
    plt.close()

    return sim_matrix


def main():
    parser = argparse.ArgumentParser(description='Null baseline bQ analysis for symmetric models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to symmetric model checkpoint')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files')
    parser.add_argument('--data_path', type=str,
                       default="/workspace/modded-nanogpt/data/finewebedu10B/finewebedu_train_000099.bin",
                       help='Path to training data')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESULTS_FILENAME = f"{args.output_prefix}_results.txt"
    TOP_K_TOKENS = 15

    with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
        log_output("=" * 60, f)
        log_output("NULL BASELINE: bQ Analysis for Symmetric Model", f)
        log_output("=" * 60, f)
        log_output(f"This shows alignment with RANDOM bQ vectors", f)
        log_output(f"to serve as a null hypothesis baseline.", f)
        log_output("=" * 60, f)
        log_output(f"Device: {DEVICE}", f)
        log_output(f"Checkpoint: {args.checkpoint}", f)

        # Load model
        model, config_obj = load_symmetric_model(args.checkpoint, DEVICE, f)

        # Load vocab
        meta_path = os.path.join(os.path.dirname(args.data_path), "meta.pkl")
        itos = load_vocab(meta_path, config_obj.vocab_size, f)

        # Generate random bQ vectors
        n_layer = config_obj.n_layer
        d_head = config_obj.n_embd // config_obj.n_head

        log_output(f"\nGenerating random bQ vectors (mean=0.5, std=[0.05, 0.15])...", f)
        bQ_random = generate_random_bQ(n_layer, d_head)

        # Plot bQ norms
        bQ_norms = torch.norm(bQ_random, p=2, dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_layer), bQ_norms)
        plt.xlabel('Layer Index')
        plt.ylabel('Norm of Random bQ')
        plt.title('Norm of Random bQ per Layer (NULL BASELINE)')
        plt.xticks(range(n_layer))
        plt.grid(True, axis='y', linestyle=':')
        save_path = f"{args.output_prefix}_bQ_mean_norm.png"
        plt.savefig(save_path)
        log_output(f"Saved: {save_path}", f)
        plt.close()

        # Run analysis
        token_results = analyze_null_alignment(
            model, bQ_random, itos, TOP_K_TOKENS, DEVICE, f
        )

        pos_results = analyze_null_positional(
            model, bQ_random, config_obj.context_length, DEVICE, args.output_prefix, f
        )

        log_output("\n\n--- NULL BASELINE Analysis Complete ---", f)
        log_output(f"Results saved to {RESULTS_FILENAME}", f)
        log_output("\nKey observation: With random bQ vectors, the top/bottom", f)
        log_output("aligned tokens should be essentially random with no", f)
        log_output("semantic or syntactic pattern.", f)


if __name__ == "__main__":
    main()
