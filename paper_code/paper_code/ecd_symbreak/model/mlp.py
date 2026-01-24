"""
MLP modules for ECD symmetry breaking experiments.

Provides:
- MLP: Standard GELU MLP (for baseline comparison)
- RandomPReLU1d: PReLU with per-feature learnable negative slopes
- AsymmetricMLPPreLU: MLP using RandomPReLU1d (default for symmetry breaking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Standard MLP with GELU activation.

    Used for symmetric baseline comparison.

    Args:
        config: GPTConfig instance
    """

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
    """
    PReLU with per-feature learnable negative slopes.

    The negative slopes are initialized from N(init_slope, slope_std),
    providing asymmetry in the activation function.

    Args:
        features: Number of features (hidden dimension)
        init_slope: Mean of initial slope distribution (default: 0.2)
        slope_std: Std of initial slope distribution (default: 1.0)
    """

    def __init__(self, features, init_slope=0.2, slope_std=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features) * slope_std + init_slope)

    def forward(self, x):
        # x: (B, T, C) -> transpose to (B, C, T) for prelu, then back
        x2 = x.transpose(1, 2)
        y2 = F.prelu(x2, self.weight)
        return y2.transpose(1, 2)


class AsymmetricMLPPreLU(nn.Module):
    """
    MLP with asymmetric PReLU activation.

    Uses RandomPReLU1d with per-feature learnable negative slopes,
    breaking the symmetry present in symmetric activations like GELU.

    This is the default MLP for symmetry-breaking experiments.

    Args:
        config: GPTConfig instance
    """

    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))
