# GELU Ablation Experiments

This directory contains isolated experiments comparing GELU MLP (standard GPT-2) against the Asymmetric PReLU MLP used in the initial paper experiments in the paper. These runs are completely separated from the prelu mlp case, to avoid contamination of results.

## Purpose

To determine whether the PReLU activation specifically contributes to the symmetry-breaking benefits observed in our experiments, or whether the attention-based symmetry breaking (bQ+bV) is sufficient.

## Directory Structure

```
gelu_ablation/
├── README.md              # This file
├── code/
│   └── train_gpt_gelu_ablation.py  # Isolated training script with GELU MLP
├── scripts/
│   ├── run_ecd_symmetric_gelu.sh   # ECD symmetric
│   ├── run_ecd_bQbV_gelu.sh        # ECD with bQ+bV
│   ├── run_adam_symmetric_gelu.sh  # AdamW symmetric
│   ├── run_adam_bQbV_gelu.sh       # AdamW with bQ+bV
│   ├── run_sgdm_symmetric_gelu.sh  # SGDM symmetric
│   ├── run_sgdm_bQbV_gelu.sh       # SGDM with bQ+bV
│   ├── run_soap_symmetric_gelu.sh  # SOAP symmetric
│   ├── run_soap_bQbV_gelu.sh       # SOAP with bQ+bV
│   └── run_all_gelu_parallel.sh    # Run all 8 experiments in parallel
└── results/               # Training outputs (created during runs)
```

## 8 Experiments

| # | Optimizer | Mode | MLP | Script |
|---|-----------|------|-----|--------|
| 1 | ECD | Symmetric | GELU | `run_ecd_symmetric_gelu.sh` |
| 2 | ECD | bQ+bV | GELU | `run_ecd_bQbV_gelu.sh` |
| 3 | AdamW | Symmetric | GELU | `run_adam_symmetric_gelu.sh` |
| 4 | AdamW | bQ+bV | GELU | `run_adam_bQbV_gelu.sh` |
| 5 | SGDM | Symmetric | GELU | `run_sgdm_symmetric_gelu.sh` |
| 6 | SGDM | bQ+bV | GELU | `run_sgdm_bQbV_gelu.sh` |
| 7 | SOAP | Symmetric | GELU | `run_soap_symmetric_gelu.sh` |
| 8 | SOAP | bQ+bV | GELU | `run_soap_bQbV_gelu.sh` |

## Configuration

All runs use **identical hyperparameters** to the paper PReLU runs:

- **Model**: 124M parameters
- **vocab_size**: 50304
- **Training**: 500M tokens, batch_size=8, context=512
- **Seed**: 42

### Optimizer Hyperparameters

| Optimizer | Hyperparameters |
|-----------|-----------------|
| **ECD** | lr=1.0, eta=100, F0=2, wd=0.0 |
| **AdamW** | lr=0.0001, wd=0.0, betas=(0.9, 0.999) |
| **SOAP** | lr=0.0003, betas=(0.95, 0.95), wd=0.01, prefreq=10 |
| **SGDM** | lr=0.03, momentum=0.95, nesterov=True, init_velocity_std=0.1 |

## Key Difference from Initial Paper Runs

**Paper runs** use `AsymmetricMLPPreLU`:
```python
class AsymmetricMLPPreLU(nn.Module):
    def __init__(self, config):
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)  # Learnable per-feature slopes
        self.c_proj = nn.Linear(hidden, config.n_embd)
```

**These ablation runs** use standard GELU MLP:
```python
class GELUMLP(nn.Module):
    def __init__(self, config):
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        # Standard GELU - no learnable parameters, symmetric function
        self.c_proj = nn.Linear(hidden, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))
```

## Running the Experiments

### Run all 8 experiments (requires 4 GPUs, runs in 2 rounds)
```bash
./scripts/run_all_gelu_parallel.sh
```

### Run only symmetric experiments (4 GPUs)
```bash
./scripts/run_all_gelu_parallel.sh symmetric
```

### Run only bQ+bV experiments (4 GPUs)
```bash
./scripts/run_all_gelu_parallel.sh bQbV
```

### Run individual experiments
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_ecd_bQbV_gelu.sh
```

## Comparison with Paper Results (PReLU)

After running these experiments, compare validation loss against paper PReLU runs:

### Paper PReLU Results (for reference)

| Optimizer | Symmetric PReLU | bQ+bV PReLU |
|-----------|-----------------|-------------|
| **ECD** | 3.7992 | ~3.60 |
| **AdamW** | 3.5412 | ~3.53 |
| **SGDM** | 3.6534 | ~3.66 |
| **SOAP** | 3.3673 | ~3.51 |

## Notes

- W&B logging goes to a separate `gelu-ablation` project to avoid contaminating paper results
- Checkpoints and logs are saved to `results/<opt>-<mode>-gelu/`
- This code is completely isolated from the paper training scripts

---

*Created: 2026-01-20*
