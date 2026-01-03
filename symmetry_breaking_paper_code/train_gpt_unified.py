#!/usr/bin/env python3
"""
Unified GPT Training Script
============================
Combines ALL features for paper experiments:
- Symmetric mode (standard attention) OR Disordered mode (bQ symmetry breaking)
- Optional bV for V-O symmetry breaking (works in both modes)
- Optional RoPE or learned position embeddings
- Optional Flash Attention
- Multiple optimizers: ECD, Adam, SGDM, SOAP

Usage Examples:
    # Symmetric baseline (standard attention, no bQ)
    python train_gpt_unified.py --symmetric --optimizer adam

    # Disordered with bQ only (default)
    python train_gpt_unified.py --optimizer ecd

    # Disordered with bQ + bV
    python train_gpt_unified.py --use_v_bias --optimizer ecd

    # Symmetric with bV only (test V-O breaking without Q-K breaking)
    python train_gpt_unified.py --symmetric --use_v_bias --optimizer adam

    # With RoPE instead of learned positions
    python train_gpt_unified.py --use_rope --optimizer ecd
"""

from dataclasses import dataclass
import os, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check Flash Attention availability
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

# --- optional logger ---
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# --- ECD optimizer ---
from ECD_q1_scaled import ECD_q1_scaled

# Try to import SOAP
try:
    from soap import SOAP
    SOAP_AVAILABLE = True
except ImportError:
    SOAP_AVAILABLE = False

# -------------------------
# W&B optimizer serialization helpers
# -------------------------
def _to_serializable(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    try:
        if torch.is_tensor(x):
            return x.item() if x.numel() == 1 else f"tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    except Exception:
        pass
    return str(x)

def serialize_optimizer(opt):
    info = {
        "class": opt.__class__.__name__,
        "defaults": {},
        "param_groups": []
    }
    d = getattr(opt, "defaults", {})
    info["defaults"] = {k: _to_serializable(v) for k, v in d.items()}
    for g in getattr(opt, "param_groups", []):
        g_copy = {k: _to_serializable(v) for k, v in g.items() if k != "params"}
        info["param_groups"].append(g_copy)
    return info

def wandb_record_optimizer(run, opt_kind, opt_obj, ecd_kwargs=None, lr_calibrated=None):
    if run is None:
        return
    info = serialize_optimizer(opt_obj)
    payload = {
        "optimizer": {
            "kind": opt_kind,
            "class": info["class"],
            "defaults": info["defaults"],
            "param_groups": info["param_groups"],
        }
    }
    if lr_calibrated is not None:
        payload["optimizer"]["calibrated_lr"] = lr_calibrated
    run.config.update(payload, allow_val_change=True)

    if opt_kind == "ecd" and ecd_kwargs is not None:
        run.config.update({
            "ecd": {
                "lr": ecd_kwargs.get("lr"),
                "F0": ecd_kwargs.get("F0"),
                "eps1": ecd_kwargs.get("eps1"),
                "eps2": ecd_kwargs.get("eps2"),
                "nu": ecd_kwargs.get("nu"),
                "weight_decay": ecd_kwargs.get("weight_decay"),
                "eta": ecd_kwargs.get("eta"),
                "consEn": ecd_kwargs.get("consEn"),
            }
        }, allow_val_change=True)

# -------------------------
# misc utils
# -------------------------
def maybe_graph_break():
    try:
        import torch._dynamo as _dynamo
        _dynamo.graph_break()
    except Exception:
        pass

def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Model config presets
# -------------------------
@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768
    rotary_base:    int = 10000
    rope_pct:       float = 0.5  # Fraction of head dim to apply RoPE to

PRESETS = {
    "124m": GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "355m": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "774m": GPTConfig(n_layer=36, n_head=20, n_embd=1280),
    "1.3b": GPTConfig(n_layer=48, n_head=25, n_embd=1600),
}

# -------------------------
# RoPE Implementation
# -------------------------
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    Handles cases where rotary dimension is smaller than head dimension.
    """
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat((q_rot, q_pass), dim=-1)
    k_embed = torch.cat((k_rot, k_pass), dim=-1)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, config: GPTConfig, device=None):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head
        rotary_dim = int(head_dim * config.rope_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (config.rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        B, T = position_ids.shape
        inv_freq = self.inv_freq.to(x.device)
        freqs = torch.einsum('i,bt->bti', inv_freq, position_ids.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

# -------------------------
# SYMMETRIC Attention (Standard - no bQ/bK)
# -------------------------
class CausalSelfAttention(nn.Module):
    """Standard causal self-attention without symmetry breaking biases."""
    def __init__(self, config, use_v_bias=False, mean_V=None, std_V=None,
                 use_rope=False, use_flash=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.use_rope = use_rope
        self.use_flash = use_flash and FLASH_AVAILABLE
        self.use_v_bias = use_v_bias

        # V-O symmetry breaking (optional, even in symmetric mode)
        if use_v_bias:
            def to_vec(x, default):
                if x is None:
                    x = default
                x = torch.as_tensor(x, dtype=torch.float32)
                if x.ndim == 0:
                    x = x.repeat(self.d)
                return x.view(self.d)
            mean_V = to_vec(mean_V, 0.0)
            std_V = to_vec(std_V, 0.05)
            self.register_buffer("mean_V", mean_V)
            self.register_buffer("std_V", std_V.abs())
            self.register_buffer("bV", torch.zeros(self.n_head, self.d))

        if not self.use_flash:
            mask = torch.tril(torch.ones(config.context_length, config.context_length))
            self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    @torch.no_grad()
    def _resample_v_bias(self, device):
        if self.use_v_bias:
            mV, sV = self.mean_V.to(device), self.std_V.to(device)
            base_V = mV + sV * torch.randn_like(mV)
            self.bV.copy_(base_V.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x, rope_cos_sin=None):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        if self.training and self.use_v_bias:
            maybe_graph_break()
            self._resample_v_bias(x.device)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1, 2)
        k = k.view(B, T, nh, d).transpose(1, 2)
        v = v.view(B, T, nh, d).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # V-O symmetry breaking (optional)
        if self.use_v_bias:
            v = v + self.bV.view(1, nh, 1, d)

        # Attention computation
        if self.use_flash:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# -------------------------
# DISORDERED Attention (bQ/bK symmetry breaking)
# -------------------------
class DisorderedCausalSelfAttention(nn.Module):
    """
    Attention with Q-K symmetry breaking (bQ, optionally bK) and optional V-O breaking (bV).
    """
    def __init__(self, config,
                 mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mean_V=None, std_V=None,
                 mode="per_batch", use_k_bias=True, use_v_bias=False,
                 use_rope=False, use_flash=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.use_rope = use_rope
        self.use_flash = use_flash and FLASH_AVAILABLE

        def to_vec(x, default):
            if x is None:
                x = default
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 0:
                x = x.repeat(self.d)
            return x.view(self.d)

        # Q-K biases
        mean_Q = to_vec(mean_Q, 0.5)
        mean_K = to_vec(mean_K, 0.3)
        std_Q = to_vec(std_Q, torch.linspace(0.05, 0.15, self.d))
        std_K = to_vec(std_K, torch.linspace(0.12, 0.08, self.d))

        # V-O biases
        mean_V = to_vec(mean_V, 0.0)
        std_V = to_vec(std_V, 0.05)

        # Register Q-K buffers
        self.register_buffer("mean_Q", mean_Q)
        self.register_buffer("mean_K", mean_K)
        self.register_buffer("std_Q", std_Q.abs())
        self.register_buffer("std_K", std_K.abs())
        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))

        # Register V-O buffers
        self.register_buffer("mean_V", mean_V)
        self.register_buffer("std_V", std_V.abs())
        self.register_buffer("bV", torch.zeros(self.n_head, self.d))

        self.mode = mode
        self.use_k_bias = use_k_bias
        self.use_v_bias = use_v_bias

        if not self.use_flash:
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

        if self.use_v_bias:
            mV, sV = self.mean_V.to(device), self.std_V.to(device)
            base_V = mV + sV * torch.randn_like(mV)
            self.bV.copy_(base_V.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x, rope_cos_sin=None):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        if self.training and self.mode == "per_batch":
            maybe_graph_break()
            self._resample_biases(x.device)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1, 2)
        k = k.view(B, T, nh, d).transpose(1, 2)
        v = v.view(B, T, nh, d).transpose(1, 2)

        # Apply RoPE BEFORE symmetry-breaking biases
        if self.use_rope and rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Q-K symmetry breaking
        q = q + self.bQ.view(1, nh, 1, d)
        if self.use_k_bias:
            k = k + self.bK.view(1, nh, 1, d)

        # V-O symmetry breaking
        if self.use_v_bias:
            v = v + self.bV.view(1, nh, 1, d)

        # Attention computation
        if self.use_flash:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# -------------------------
# MLP variants
# -------------------------
class MLP(nn.Module):
    """Standard MLP with GELU activation (for symmetric mode)."""
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class RandomPReLU1d(nn.Module):
    def __init__(self, features, init_slope=0.2, slope_std=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features) * slope_std + init_slope)
    def forward(self, x):
        x2 = x.transpose(1, 2)
        y2 = F.prelu(x2, self.weight)
        return y2.transpose(1, 2)

class AsymmetricMLPPreLU(nn.Module):
    """MLP with asymmetric PReLU activation (for disordered mode)."""
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

# -------------------------
# Block variants
# -------------------------
class SymmetricBlock(nn.Module):
    """Block with standard attention and GELU MLP."""
    def __init__(self, config, use_v_bias=False, mean_V=None, std_V=None,
                 use_rope=False, use_flash=False):
        super().__init__()
        self.sa = CausalSelfAttention(
            config, use_v_bias=use_v_bias, mean_V=mean_V, std_V=std_V,
            use_rope=use_rope, use_flash=use_flash
        )
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, rope_cos_sin=None):
        x = x + self.sa(self.ln1(x), rope_cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class DisorderedBlock(nn.Module):
    """Block with disordered attention and asymmetric MLP."""
    def __init__(self, config, use_k_bias=True, use_v_bias=False,
                 mean_V=None, std_V=None, use_rope=False, use_flash=False):
        super().__init__()
        self.sa = DisorderedCausalSelfAttention(
            config, mode="per_batch",
            use_k_bias=use_k_bias, use_v_bias=use_v_bias,
            mean_V=mean_V, std_V=std_V,
            use_rope=use_rope, use_flash=use_flash
        )
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, rope_cos_sin=None):
        x = x + self.sa(self.ln1(x), rope_cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x

# -------------------------
# Unified GPT Model
# -------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig, symmetric=False,
                 use_k_bias=True, use_v_bias=False,
                 mean_V=None, std_V=None,
                 use_rope=False, use_flash=False):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        self.symmetric = symmetric

        # Select block type based on mode
        if symmetric:
            BlockClass = SymmetricBlock
            block_kwargs = dict(
                use_v_bias=use_v_bias, mean_V=mean_V, std_V=std_V,
                use_rope=use_rope, use_flash=use_flash
            )
        else:
            BlockClass = DisorderedBlock
            block_kwargs = dict(
                use_k_bias=use_k_bias, use_v_bias=use_v_bias,
                mean_V=mean_V, std_V=std_V,
                use_rope=use_rope, use_flash=use_flash
            )

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([BlockClass(config, **block_kwargs) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Position embeddings: learned (default) or RoPE
        if not use_rope:
            self.transformer['wpe'] = nn.Embedding(config.context_length, config.n_embd)
        else:
            self.rope_emb = RotaryEmbedding(config)

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

        x = self.transformer.wte(idx)

        if self.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            rope_cos_sin = self.rope_emb(x, pos)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
            rope_cos_sin = None

        for block in self.transformer.h:
            x = block(x, rope_cos_sin)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------------------
# Robust FineWeb loader
# -------------------------
def load_tokens_autodetect(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".bin":
        with open(filename, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            if header[0] == 20240520:
                num_tokens = header[2]
                tokens = np.fromfile(f, dtype=np.uint16, count=num_tokens)
                return torch.from_numpy(tokens.astype(np.int64))
            else:
                f.seek(0)
                arr = np.fromfile(f, dtype=np.uint16)
                return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if ext == ".npy":
        arr = np.load(filename)
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if ext in (".pt", ".pth"):
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(np.asarray(t).astype(np.int64), dtype=torch.long)
    raise RuntimeError(f"Unknown shard extension: {ext}")

def load_tokens_with_hint(filename, data_ext):
    if data_ext in ("auto", "bin"):
        return load_tokens_autodetect(filename)
    if data_ext == "npy":
        arr = np.load(filename)
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if data_ext == "pt":
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(np.asarray(t).astype(np.int64), dtype=torch.long)
    raise RuntimeError(f"Bad --data_ext: {data_ext}")

class FineWebShards:
    def __init__(self, data_root, split, B, T, process_rank=0, num_processes=1,
                 data_ext="auto", vocab_size=None, sanity_check=False):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_ext = data_ext
        self.vocab_size = vocab_size
        self.sanity_check = sanity_check
        shards = sorted([os.path.join(data_root, s)
                         for s in os.listdir(data_root)
                         if (split in s and (data_ext=="auto" or s.endswith("."+data_ext)))])
        if data_ext == "auto":
            shards = [s for s in shards if os.path.splitext(s)[1].lower() in (".bin",".npy",".pt",".pth")]
        if not shards:
            raise FileNotFoundError(f"No shards for split={split} under {data_root} (ext={data_ext})")
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens_with_hint(self.shards[self.current_shard], self.data_ext)
        if self.sanity_check and self.vocab_size is not None:
            mx = int(self.tokens.max().item())
            if mx >= self.vocab_size:
                raise ValueError(f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                                 f">= vocab_size {self.vocab_size}. Wrong file format or vocab.")
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.num_processes
        if self.pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens_with_hint(self.shards[self.current_shard], self.data_ext)
            if self.sanity_check and self.vocab_size is not None:
                mx = int(self.tokens.max().item())
                if mx >= self.vocab_size:
                    raise ValueError(f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                                     f">= vocab_size {self.vocab_size}. Wrong file format or vocab.")
            self.pos = B * T * self.process_rank
        return x, y

# -------------------------
# Optimizers
# -------------------------
def build_optimizer(model, kind, ecd_kwargs=None, adamw_kwargs=None, sgdm_kwargs=None, soap_kwargs=None):
    kind = kind.lower()
    if kind == "ecd":
        return ECD_q1_scaled(model.parameters(), **(ecd_kwargs or {}))
    if kind == "adam":
        kw = dict(weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, fused=True)
        kw.update(adamw_kwargs or {})
        return torch.optim.AdamW(model.parameters(), **kw)
    if kind == "sgdm":
        kw = dict(lr=0.03, momentum=0.95, nesterov=True, weight_decay=0.0)
        kw.update(sgdm_kwargs or {})
        return torch.optim.SGD(model.parameters(), **kw)
    if kind == "soap":
        if not SOAP_AVAILABLE:
            raise ImportError("SOAP not available. Install from soap.py")
        kw = dict(lr=3e-4, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)
        kw.update(soap_kwargs or {})
        return SOAP(model.parameters(), **kw)
    raise ValueError(f"Unknown optimizer {kind}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Unified GPT training with symmetric/disordered modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Symmetric baseline (standard attention, no bQ)
  python train_gpt_unified.py --symmetric --optimizer adam

  # Disordered with bQ (default)
  python train_gpt_unified.py --optimizer ecd

  # Disordered with bQ + bV
  python train_gpt_unified.py --use_v_bias --optimizer ecd

  # Symmetric with bV only
  python train_gpt_unified.py --symmetric --use_v_bias --optimizer adam
"""
    )

    # Model architecture
    ap.add_argument("--model", choices=PRESETS.keys(), default="124m")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=50304)

    # Training parameters
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--train_tokens", type=float, default=5e8)
    ap.add_argument("--valid_every_updates", type=int, default=1000)

    # Optimizer selection
    ap.add_argument("--optimizer", choices=["ecd", "adam", "sgdm", "soap"], default="ecd")

    # MODE: symmetric vs disordered
    ap.add_argument("--symmetric", action="store_true", default=False,
                    help="Use symmetric architecture (standard attention, no bQ). "
                         "Default is disordered (bQ symmetry breaking).")

    # ECD hypers
    ap.add_argument("--ecd_lr", type=float, default=0.5)
    ap.add_argument("--ecd_eta", type=float, default=100.0)
    ap.add_argument("--ecd_F0", type=float, default=2.0)
    ap.add_argument("--ecd_eps1", type=float, default=1e-10)
    ap.add_argument("--ecd_eps2", type=float, default=1e-40)
    ap.add_argument("--ecd_nu", type=float, default=0.0)
    ap.add_argument("--ecd_wd", type=float, default=0.0)
    ap.add_argument("--ecd_consEn", action="store_true")
    ap.add_argument("--no_ecd_consEn", dest="ecd_consEn", action="store_false")
    ap.set_defaults(ecd_consEn=True)

    # AdamW hypers
    ap.add_argument("--adam_lr", type=float, default=1e-4)
    ap.add_argument("--adam_wd", type=float, default=0.01)

    # SGDM hypers
    ap.add_argument("--sgdm_lr", type=float, default=0.03)
    ap.add_argument("--sgdm_momentum", type=float, default=0.95)
    ap.add_argument("--sgdm_nesterov", action="store_true", default=True)

    # SOAP hypers
    ap.add_argument("--soap_lr", type=float, default=3e-4)
    ap.add_argument("--soap_beta1", type=float, default=0.95)
    ap.add_argument("--soap_beta2", type=float, default=0.95)
    ap.add_argument("--soap_wd", type=float, default=0.01)
    ap.add_argument("--soap_prefreq", type=int, default=10)

    # Data loader options
    ap.add_argument("--data_ext", choices=["auto", "bin", "npy", "pt"], default="auto")
    ap.add_argument("--sanity_check_tokens", action="store_true")

    # Bias controls - Q-K sector (only relevant if not --symmetric)
    ap.add_argument("--disable_k_bias", action="store_true",
                    help="Use only bQ; set bK=0 (only applies in disordered mode)")

    # Bias controls - V-O sector (works in both symmetric and disordered modes)
    ap.add_argument("--use_v_bias", action="store_true", default=False,
                    help="Enable bV for V-O symmetry breaking")
    ap.add_argument("--mean_V", type=float, default=0.0)
    ap.add_argument("--std_V", type=float, default=0.05)

    # RoPE and Flash Attention
    ap.add_argument("--use_rope", action="store_true", default=False,
                    help="Use Rotary Position Embeddings instead of learned position embeddings")
    ap.add_argument("--use_flash", action="store_true", default=False,
                    help="Use Flash Attention (requires compatible GPU)")

    # Run configuration
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="/workspace/modded-nanogpt/data/finewebedu10B")
    ap.add_argument("--log_dir", type=str, default="runs")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--save_every_updates", type=int, default=0)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--wandb", action="store_true")

    args = ap.parse_args()

    # Validate flash attention availability
    if args.use_flash and not FLASH_AVAILABLE:
        print("Warning: Flash Attention requested but not available. Falling back to standard attention.")
        args.use_flash = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    cfg = PRESETS[args.model]
    cfg.context_length = args.ctx
    cfg.vocab_size = args.vocab
    device = get_device_type()
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32
    torch.set_float32_matmul_precision("high")

    # Build model
    model = GPT(
        cfg,
        symmetric=args.symmetric,
        use_k_bias=(not args.disable_k_bias),
        use_v_bias=args.use_v_bias,
        mean_V=args.mean_V,
        std_V=args.std_V,
        use_rope=args.use_rope,
        use_flash=args.use_flash
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Print configuration
    mode_str = "SYMMETRIC" if args.symmetric else "DISORDERED"
    print(f"=== Configuration ===")
    print(f"  Mode: {mode_str}")
    print(f"  Model: {args.model} ({n_params:,} params)")
    print(f"  Position encoding: {'RoPE' if args.use_rope else 'Learned'}")
    print(f"  Attention: {'Flash' if args.use_flash else 'Standard'}")
    if args.symmetric:
        print(f"  Q-K sector: bQ=disabled (symmetric)")
    else:
        print(f"  Q-K sector: bQ=enabled, bK={'enabled' if not args.disable_k_bias else 'disabled'}")
    print(f"  V-O sector: bV={'enabled' if args.use_v_bias else 'disabled'}")
    if args.use_v_bias:
        print(f"    mean_V={args.mean_V}, std_V={args.std_V}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"=====================")

    # Data
    T = cfg.context_length
    B = args.batch_size
    gdup = args.grad_accum
    train_loader = FineWebShards(args.data_root, "train", B=B, T=T,
                                 data_ext=args.data_ext, vocab_size=cfg.vocab_size,
                                 sanity_check=args.sanity_check_tokens)
    val_loader = FineWebShards(args.data_root, "val", B=B, T=T,
                               data_ext=args.data_ext, vocab_size=cfg.vocab_size,
                               sanity_check=args.sanity_check_tokens)

    # W&B setup - build descriptive tags
    mode_tag = "symmetric" if args.symmetric else "disorder"
    symbreak_tag = "bQ" if not args.symmetric else "nobQ"
    if args.use_v_bias:
        symbreak_tag += "+bV"
    pos_tag = "rope" if args.use_rope else "learned"

    run_name = args.name or f"{args.model}-{args.optimizer}-{mode_tag}-{symbreak_tag}-{pos_tag}-seed{args.seed}"
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.wandb and WANDB_AVAILABLE:
        tags = [args.optimizer, args.model, mode_tag, pos_tag]
        if args.use_v_bias:
            tags.append("bV")
        wandb.init(
            project="ecd-unified",
            name=run_name,
            tags=tags,
            config={
                "dataset": "finewebedu10B",
                "model": vars(cfg),
                "params": n_params,
                "train_tokens": args.train_tokens,
                "opt_kind": args.optimizer,
                "mode": mode_tag,
                "position_encoding": "rope" if args.use_rope else "learned",
                "attention": "flash" if args.use_flash else "standard",
                "symmetry_breaking": {
                    "symmetric": args.symmetric,
                    "bQ": not args.symmetric,
                    "bK": (not args.symmetric) and (not args.disable_k_bias),
                    "bV": args.use_v_bias,
                    "mean_V": args.mean_V if args.use_v_bias else None,
                    "std_V": args.std_V if args.use_v_bias else None,
                }
            }
        )

    # Build optimizer
    if args.optimizer == "ecd":
        ecd_kwargs = dict(lr=args.ecd_lr, F0=args.ecd_F0, eps1=args.ecd_eps1,
                          eps2=args.ecd_eps2, nu=args.ecd_nu, weight_decay=args.ecd_wd,
                          eta=args.ecd_eta, consEn=args.ecd_consEn)
        opt = build_optimizer(model, "ecd", ecd_kwargs=ecd_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "ecd", opt, ecd_kwargs=ecd_kwargs)
    elif args.optimizer == "adam":
        adamw_kwargs = dict(lr=args.adam_lr, weight_decay=args.adam_wd, betas=(0.9, 0.95))
        opt = build_optimizer(model, "adam", adamw_kwargs=adamw_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "adam", opt)
    elif args.optimizer == "sgdm":
        sgdm_kwargs = dict(lr=args.sgdm_lr, momentum=args.sgdm_momentum, nesterov=args.sgdm_nesterov)
        opt = build_optimizer(model, "sgdm", sgdm_kwargs=sgdm_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "sgdm", opt)
    elif args.optimizer == "soap":
        soap_kwargs = dict(lr=args.soap_lr, betas=(args.soap_beta1, args.soap_beta2),
                           weight_decay=args.soap_wd, precondition_frequency=args.soap_prefreq)
        opt = build_optimizer(model, "soap", soap_kwargs=soap_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "soap", opt)

    # Training loop
    consumed_tokens = 0
    update = 0
    best_val_loss = float('inf')
    model.train()
    start = time.time()

    while consumed_tokens < args.train_tokens:
        t0 = time.time()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for _ in range(gdup):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(get_device_type(), enabled=(dtype == torch.bfloat16), dtype=dtype):
                _, loss = model(x, y)
            (loss / gdup).backward()
            total_loss += float(loss.detach().item())
            consumed_tokens += (B * T)

        avg_loss = torch.tensor(total_loss / gdup, device=device)

        def closure():
            return avg_loss

        try:
            opt.step(closure)
        except TypeError:
            opt.step()

        update += 1
        dt = time.time() - t0
        tok_per_s = (B * T * gdup) / max(dt, 1e-9)

        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"train/loss": avg_loss.item(),
                       "speed/tok_per_s": tok_per_s,
                       "train/consumed_tokens": consumed_tokens,
                       "update": update})

        if update % max(1, args.valid_every_updates) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = val_loader.next_batch()
                vx, vy = vx.to(device), vy.to(device)
                with torch.autocast(get_device_type(), enabled=(dtype == torch.bfloat16), dtype=dtype):
                    _, vloss = model(vx, vy)
            val_loss = vloss.item()
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"val/loss": val_loss, "update": update})
            print(f"[up {update:6d}] loss {avg_loss.item():.4f} | val {val_loss:.4f} "
                  f"| tok/s {tok_per_s:,.0f} | consumed {int(consumed_tokens):,}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "model_best.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "update": update,
                    "consumed_tokens": consumed_tokens,
                    "val_loss": val_loss,
                    "symmetric": args.symmetric,
                    "use_rope": args.use_rope,
                    "use_flash": args.use_flash,
                    "symmetry_breaking": {
                        "symmetric": args.symmetric,
                        "bQ": not args.symmetric,
                        "bK": (not args.symmetric) and (not args.disable_k_bias),
                        "bV": args.use_v_bias,
                        "mean_V": args.mean_V,
                        "std_V": args.std_V
                    }
                }, best_path)
                print(f"  -> New best! Saved to {best_path}")

            model.train()

        if args.save_every_updates and (update % args.save_every_updates == 0):
            path = os.path.join(ckpt_dir, f"model_{update:06d}.pt")
            torch.save({"model": model.state_dict(), "config": cfg, "update": update,
                        "consumed_tokens": consumed_tokens}, path)

    # Final save
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "update": update,
        "consumed_tokens": consumed_tokens,
        "best_val_loss": best_val_loss,
        "symmetric": args.symmetric,
        "use_rope": args.use_rope,
        "use_flash": args.use_flash,
        "symmetry_breaking": {
            "symmetric": args.symmetric,
            "bQ": not args.symmetric,
            "bK": (not args.symmetric) and (not args.disable_k_bias),
            "bV": args.use_v_bias,
            "mean_V": args.mean_V,
            "std_V": args.std_V
        }
    }, final_path)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {final_path}. Total time: {time.time()-start:.1f}s")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
