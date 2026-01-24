"""
GPT model for ECD symmetry breaking experiments.

Supports both symmetric and disordered (symmetry-breaking) modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GPTConfig
from .block import SymmetricBlock, DisorderedBlock


class GPT(nn.Module):
    """
    GPT model with configurable symmetric/disordered modes.

    Args:
        config: GPTConfig instance
        symmetric: Use symmetric mode (no bQ) if True, disordered if False
        use_k_bias: Enable bK in disordered mode
        use_v_bias: Enable bV for V-O symmetry breaking
        mean_V: Mean of bV distribution
        std_V: Std of bV distribution
        use_prelu: Use PReLU MLP in symmetric mode (for architecture matching)
    """

    def __init__(self, config: GPTConfig, symmetric=False,
                 use_k_bias=True, use_v_bias=True,
                 mean_V=None, std_V=None,
                 use_prelu=False):
        super().__init__()
        self.config = config
        self.symmetric = symmetric

        # Select block type based on mode
        if symmetric:
            blocks = [
                SymmetricBlock(
                    config, use_v_bias=use_v_bias,
                    mean_V=mean_V, std_V=std_V,
                    use_prelu=use_prelu
                )
                for _ in range(config.n_layer)
            ]
        else:
            blocks = [
                DisorderedBlock(
                    config, use_k_bias=use_k_bias, use_v_bias=use_v_bias,
                    mean_V=mean_V, std_V=std_V
                )
                for _ in range(config.n_layer)
            ]

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.context_length, config.n_embd),
            h=nn.ModuleList(blocks),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with NanoGPT-style scaling."""
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
        """
        Forward pass.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T), optional

        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
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
