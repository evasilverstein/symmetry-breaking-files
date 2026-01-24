"""
Transformer block modules for ECD symmetry breaking experiments.

Provides:
- SymmetricBlock: CausalSelfAttention + MLP (GELU or PReLU)
- DisorderedBlock: FullSymmetryBrokenAttention + AsymmetricMLPPreLU
"""

import torch.nn as nn

from .attention import CausalSelfAttention, FullSymmetryBrokenAttention
from .mlp import MLP, AsymmetricMLPPreLU


class SymmetricBlock(nn.Module):
    """
    Transformer block with standard attention (no bQ).

    Can use either GELU (default) or PReLU MLP.  **In the main paper examples we used PreLU**

    Args:
        config: GPTConfig instance
        use_v_bias: Enable bV for V-O symmetry breaking (optional)
        mean_V: Mean of bV distribution
        std_V: Std of bV distribution
        use_prelu: Use PReLU MLP instead of GELU (for architecture matching)
    """

    def __init__(self, config, use_v_bias=False, mean_V=None, std_V=None,
                 use_prelu=False):
        super().__init__()
        self.sa = CausalSelfAttention(
            config, use_v_bias=use_v_bias, mean_V=mean_V, std_V=std_V
        )
        # Use PReLU MLP to match disordered architecture if requested
        self.mlp = AsymmetricMLPPreLU(config) if use_prelu else MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DisorderedBlock(nn.Module):
    """
    Transformer block with full symmetry breaking.

    Uses FullSymmetryBrokenAttention (bQ + optional bV) and AsymmetricMLPPreLU.

    Args:
        config: GPTConfig instance
        use_k_bias: Enable bK (default: True, but often disabled)
        use_v_bias: Enable bV for V-O symmetry breaking
        mean_V: Mean of bV distribution
        std_V: Std of bV distribution
    """

    def __init__(self, config, use_k_bias=True, use_v_bias=True,
                 mean_V=None, std_V=None):
        super().__init__()
        self.sa = FullSymmetryBrokenAttention(
            config, mode="per_batch",
            use_k_bias=use_k_bias, use_v_bias=use_v_bias,
            mean_V=mean_V, std_V=std_V
        )
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
