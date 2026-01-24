"""
ECD Symmetry Breaking Package

This package implements the ECD optimizer with symmetry-breaking mechanisms
for transformer training, as described in the accompanying paper.

Key Components:
- config: GPTConfig and model presets
- optimizer: ECD_q1_scaled optimizer
- model: GPT model with symmetric/disordered attention variants
- data: FineWeb data loader
- utils: Helper utilities
"""

from .config import GPTConfig, PRESETS
from .optimizer import ECD_q1_scaled
from .utils import maybe_graph_break, serialize_optimizer, wandb_record_optimizer

__version__ = "1.0.0"
__all__ = [
    "GPTConfig",
    "PRESETS",
    "ECD_q1_scaled",
    "maybe_graph_break",
    "serialize_optimizer",
    "wandb_record_optimizer",
]
