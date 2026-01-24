"""
GPT Configuration and Model Presets

Default vocab_size=50304 matches the tested configuration from seed 789/83 runs.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    context_length: int = 512
    vocab_size: int = 50304  # Default matches tested runs
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# Model size presets
PRESETS = {
    "124m": GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "355m": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "774m": GPTConfig(n_layer=36, n_head=20, n_embd=1280),
    "1.3b": GPTConfig(n_layer=48, n_head=25, n_embd=1600),
}
