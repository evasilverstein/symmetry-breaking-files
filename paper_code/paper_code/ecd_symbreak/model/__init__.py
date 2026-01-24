"""
Model components for ECD symmetry breaking experiments.

Provides:
- CausalSelfAttention: Standard attention (symmetric mode), optional bV only
- FullSymmetryBrokenAttention: Full symmetry breaking with bQ + bV biases
- CausalSelfAttentionWithBias: Inference-time attention with fixed biases
- MLP: Standard GELU MLP (for baseline comparison)
- RandomPReLU1d: PReLU with per-feature learnable negative slopes
- AsymmetricMLPPreLU: MLP using RandomPReLU1d (default for symmetry breaking)
- SymmetricBlock: CausalSelfAttention + MLP
- DisorderedBlock: FullSymmetryBrokenAttention + AsymmetricMLPPreLU
- GPT: Full model with mode selection
"""

from .attention import (
    CausalSelfAttention,
    FullSymmetryBrokenAttention,
    CausalSelfAttentionWithBias,
)
from .mlp import MLP, RandomPReLU1d, AsymmetricMLPPreLU
from .block import SymmetricBlock, DisorderedBlock
from .gpt import GPT

__all__ = [
    "CausalSelfAttention",
    "FullSymmetryBrokenAttention",
    "CausalSelfAttentionWithBias",
    "MLP",
    "RandomPReLU1d",
    "AsymmetricMLPPreLU",
    "SymmetricBlock",
    "DisorderedBlock",
    "GPT",
]
