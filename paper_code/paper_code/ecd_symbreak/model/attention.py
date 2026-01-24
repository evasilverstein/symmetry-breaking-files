"""
Attention modules for ECD symmetry breaking experiments.

Provides three attention variants:
1. CausalSelfAttention: Standard attention (symmetric mode), optional bV only
2. FullSymmetryBrokenAttention: Full symmetry breaking with bQ + bV biases
3. CausalSelfAttentionWithBias: Inference-time attention with fixed biases
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import maybe_graph_break


class CausalSelfAttention(nn.Module):
    """
    Standard causal self-attention (symmetric mode).

    Optionally supports bV for V-O symmetry breaking without Q-K breaking.

    Args:
        config: GPTConfig instance
        use_v_bias: Enable bV for V-O symmetry breaking
        mean_V: Mean of bV distribution (default: 0.0)
        std_V: Std of bV distribution (default: 0.05)
    """

    def __init__(self, config, use_v_bias=False, mean_V=None, std_V=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.use_v_bias = use_v_bias

        # V-O symmetry breaking (optional)
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

        # Causal mask
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    @torch.no_grad()
    def _resample_v_bias(self, device):
        if self.use_v_bias:
            mV, sV = self.mean_V.to(device), self.std_V.to(device)
            base_V = mV + sV * torch.randn_like(mV)
            self.bV.copy_(base_V.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x):
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

        # V-O symmetry breaking (optional)
        if self.use_v_bias:
            v = v + self.bV.view(1, nh, 1, d)

        # Attention computation
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FullSymmetryBrokenAttention(nn.Module):
    """
    Attention with FULL symmetry breaking: bQ (Q-K sector) + bV (V-O sector).

    The Q-K sector has O(d_k) rotational symmetry: Q -> RQ, K -> RK.
    bQ breaks this symmetry with exponential effect via softmax.

    The V-O sector has O(d_v) rotational symmetry: V -> VR, O -> R^T O.
    bV breaks this with power-law effect (linear transform).

    Note: bK is intentionally omitted as it cancels out in softmax normalization.

    Args:
        config: GPTConfig instance
        mean_Q: Mean of bQ distribution (default: 0.5)
        mean_K: Mean of bK distribution (default: 0.3, used if use_k_bias=True)
        std_Q: Std of bQ distribution
        std_K: Std of bK distribution
        mean_V: Mean of bV distribution (default: 0.0)
        std_V: Std of bV distribution (default: 0.05)
        mode: Bias resampling mode ("per_batch" or "fixed")
        use_k_bias: Enable bK (default: True, but often disabled)
        use_v_bias: Enable bV for V-O symmetry breaking (default: True)
    """

    def __init__(self, config,
                 mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mean_V=None, std_V=None,
                 mode="per_batch", use_k_bias=True, use_v_bias=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # V-O biases (smaller magnitude since effect is power-law not exponential)
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

        # Causal mask
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    @torch.no_grad()
    def _resample_biases(self, device):
        # Q-K biases
        mQ, sQ = self.mean_Q.to(device), self.std_Q.to(device)
        mK, sK = self.mean_K.to(device), self.std_K.to(device)
        base_Q = mQ + sQ * torch.randn_like(mQ)
        base_K = mK + sK * torch.randn_like(mK)
        self.bQ.copy_(base_Q.unsqueeze(0).expand(self.n_head, -1).contiguous())
        self.bK.copy_(base_K.unsqueeze(0).expand(self.n_head, -1).contiguous())

        # V-O bias
        if self.use_v_bias:
            mV, sV = self.mean_V.to(device), self.std_V.to(device)
            base_V = mV + sV * torch.randn_like(mV)
            self.bV.copy_(base_V.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x):
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

        # Q-K symmetry breaking
        q = q + self.bQ.view(1, nh, 1, d)
        if self.use_k_bias:
            k = k + self.bK.view(1, nh, 1, d)

        # V-O symmetry breaking
        if self.use_v_bias:
            v = v + self.bV.view(1, nh, 1, d)

        # Attention computation
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class CausalSelfAttentionWithBias(nn.Module):
    """
    Attention with fixed bQ/bK/bV biases applied at inference.

    Used for evaluation of trained models with specific bias values.

    Args:
        config: GPTConfig instance
        bQ: Query bias tensor (optional)
        bK: Key bias tensor (optional)
        bV: Value bias tensor (optional)
    """

    def __init__(self, config, bQ=None, bK=None, bV=None):
        super().__init__()
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

        # Handle bQ
        if bQ is not None:
            if bQ.dim() == 1:
                bQ = bQ.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bQ.dim() == 2:
                bQ = bQ.view(1, self.n_head, 1, self.d)
            self.register_buffer("bQ", bQ)
        else:
            self.register_buffer("bQ", torch.zeros(1, self.n_head, 1, self.d))

        # Handle bK
        if bK is not None:
            if bK.dim() == 1:
                bK = bK.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bK.dim() == 2:
                bK = bK.view(1, self.n_head, 1, self.d)
            self.register_buffer("bK", bK)
        else:
            self.register_buffer("bK", torch.zeros(1, self.n_head, 1, self.d))

        # Handle bV
        if bV is not None:
            if bV.dim() == 1:
                bV = bV.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bV.dim() == 2:
                bV = bV.view(1, self.n_head, 1, self.d)
            self.register_buffer("bV", bV)
        else:
            self.register_buffer("bV", torch.zeros(1, self.n_head, 1, self.d))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d).transpose(1, 2)

        q = q + self.bQ
        k = k + self.bK
        v = v + self.bV

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
