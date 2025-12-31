# Symmetry Breaking in Transformer Optimization

Code accompanying the paper on rotational symmetry breaking in transformer attention for improved optimization with energy-conserving descent (ECD).

## Overview

This repository provides training and analysis code for studying how breaking the O(d_k) rotational symmetry in transformer attention affects optimization dynamics, particularly for the physics-inspired ECD optimizer.

### Key Insight

Standard transformer attention has continuous rotational symmetries:
```
W_Q → R·W_Q,  W_K → R·W_K  (leaves attention scores invariant)
```

For Hamiltonian optimizers like ECD, these symmetries create conserved angular momenta (Noether currents) that constrain optimization dynamics. Breaking these symmetries via random query biases enables ECD to match or exceed Adam's performance.

## Repository Structure

```
symmetry_breaking_paper_code/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train_gpt_disorder.py        # Training with disordered (symmetry-broken) attention
├── train_gpt_symmetric.py       # Training with symmetric (standard) attention
├── optimizers/
│   ├── __init__.py
│   ├── ECD_q1_scaled.py         # Energy-Conserving Descent optimizer
│   └── soap.py                  # SOAP optimizer (second-order method)
├── data/
│   ├── __init__.py
│   └── fineweb.py               # FineWeb dataset preparation
├── analysis/
│   ├── __init__.py
│   ├── analysis_bQ_batch.py     # bQ alignment analysis
│   ├── analysis_bQ_null_baseline.py  # Null baseline for bQ analysis
│   └── eval_logic_puzzles.py    # Downstream task evaluation
└── examples/
    └── run_experiments.sh       # Example training commands
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch numpy tiktoken datasets tqdm matplotlib pyyaml

# Optional: Weights & Biases for experiment tracking
pip install wandb

# For bQ analysis visualization
pip install seaborn scipy
```

### Tested Environment

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with 24GB+ VRAM (for 124M model)

## Data Preparation

We use the FineWeb-Edu dataset (10B tokens). To prepare the data:

```bash
cd data
python fineweb.py -v 10B
```

This downloads and tokenizes the dataset, creating binary shards in `data/fineweb10B/`.

The data format is:
- 256 int32 header followed by uint16 tokens
- GPT-2 tokenizer (50257 vocab, padded to 50304 for efficiency)

## Training

### Quick Start

**Train with ECD + Disordered Attention (recommended):**
```bash
python train_gpt_disorder.py \
    --model 124m \
    --optimizer ecd \
    --ecd_lr 1.0 \
    --ecd_eta 100 \
    --ecd_F0 2 \
    --train_tokens 500000000 \
    --data_root data/fineweb10B \
    --wandb \
    --name ecd-disordered-run1
```

**Train with ECD + Symmetric Attention (baseline):**
```bash
python train_gpt_symmetric.py \
    --model 124m \
    --optimizer ecd \
    --ecd_lr 1.0 \
    --ecd_eta 100 \
    --ecd_F0 2 \
    --train_tokens 500000000 \
    --data_root data/fineweb10B \
    --wandb \
    --name ecd-symmetric-run1
```

**Train with AdamW + Disordered Attention:**
```bash
python train_gpt_disorder.py \
    --model 124m \
    --optimizer adam \
    --adam_lr 0.0001 \
    --adam_wd 0.01 \
    --train_tokens 500000000 \
    --data_root data/fineweb10B \
    --wandb \
    --name adamw-disordered-run1
```

### Example Script

See `examples/run_experiments.sh` for a complete set of training commands covering all optimizer/architecture combinations.

## Architecture Details

### Disordered Attention (Symmetry-Broken)

The key modification is adding random, non-learned query biases:

```python
# In DisorderedCausalSelfAttention
def forward(self, x):
    # Resample biases each batch
    bQ = mean_Q + std_Q * randn()  # Random per batch

    q = W_Q @ x + bQ  # Add bias to queries
    k = W_K @ x       # Keys unchanged (or + bK)

    # Standard attention computation
    scores = (q @ k.T) / sqrt(d_k)
    ...
```

**Key design choices:**
- `bQ` is **not learned** - if learned, symmetry persists
- Resampled **every batch** - ensures full symmetry breaking
- **Nonzero mean** empirically works better than zero-mean

### Symmetric Attention (Standard)

Standard GPT-2 attention without any biases:

```python
def forward(self, x):
    q = W_Q @ x
    k = W_K @ x
    v = W_V @ x

    scores = (q @ k.T) / sqrt(d_k)
    ...
```

## Optimizer Details

### ECD (Energy-Conserving Descent)

Physics-inspired optimizer based on Hamiltonian dynamics with energy conservation.

**Key parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `lr` (Δt̂) | Rescaled learning rate | 0.1 - 1.0 |
| `eta` (η) | Concentration parameter | 100 - 1000 |
| `F0` | Loss offset | -1 to 2 |
| `nu` (ν) | Bounce amplitude | 0.0 |
| `weight_decay` | L2 regularization | 0.0 |

**Physics intuition:**
- The Liouville measure concentrates results: p(Θ) ∝ (F - F0)^(-ηd/2)
- Higher η → tighter concentration → less variance across runs
- F0 sets the loss reference: typically F0 = F_min - 1

### AdamW

Standard AdamW with typical language model hyperparameters:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `lr` | Learning rate | 1e-4 to 6e-4 |
| `betas` | Momentum coefficients | (0.9, 0.95) |
| `weight_decay` | L2 regularization | 0.01 - 0.1 |
| `eps` | Numerical stability | 1e-8 |

### SOAP

Second-order optimizer with Shampoo-style preconditioning:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `lr` | Learning rate | 3e-3 |
| `betas` | Momentum coefficients | (0.95, 0.95) |
| `precondition_frequency` | Preconditioner update frequency | 10 |
| `weight_decay` | L2 regularization | 0.01 |

## Model Sizes

Available model configurations:

| Model | Layers | Heads | Embedding Dim | Parameters |
|-------|--------|-------|---------------|------------|
| 124m  | 12     | 12    | 768           | ~124M      |
| 355m  | 24     | 16    | 1024          | ~355M      |
| 774m  | 36     | 20    | 1280          | ~774M      |
| 1.3b  | 48     | 25    | 1600          | ~1.3B      |

Select with `--model 124m` (or 355m, 774m, 1.3b).

## bQ Alignment Analysis

After training, analyze how the model uses the preferred direction ⟨bQ⟩:

```bash
python analysis/analysis_bQ_batch.py \
    --checkpoint path/to/checkpoint.pt \
    --output_prefix results/my_analysis
```

This generates:
1. **Token alignment plots**: Which tokens align with/against ⟨bQ⟩
2. **Positional alignment plots**: Recency bias patterns
3. **Layer-wise analysis**: How alignment patterns differ across layers
4. **Quantitative metrics**: Alignment statistics saved to JSON

### Understanding the Outputs

**Token alignment (cosine similarity of W_K·x with ⟨bQ⟩):**
- High alignment → token receives enhanced attention
- Low alignment → token receives suppressed attention
- Typically: punctuation/structure tokens align positively, content words align negatively

**Positional alignment:**
- Layer 0 often shows "recency bias": later positions align more with ⟨bQ⟩
- This enhances attention to recent context

## Downstream Evaluation

Test trained models on logic puzzles:

```bash
python analysis/eval_logic_puzzles.py
```

Edit the `CHECKPOINTS` dictionary in the script to point to your trained models.

Tasks include:
- Pattern completion (numeric/alphabetic sequences)
- Retrieval (near/far context)
- Simple inference
- Negation understanding
- Syntax sensitivity

## Common Command-Line Arguments

### Training Scripts

| Argument | Description |
|----------|-------------|
| `--model` | Model size (124m, 355m, 774m, 1.3b) |
| `--optimizer` | Optimizer (ecd, adam, soap, sgdm) |
| `--train_tokens` | Total tokens to train on |
| `--batch_size` | Batch size per step |
| `--grad_accum` | Gradient accumulation steps |
| `--ctx` | Context length (default: 512) |
| `--data_root` | Path to data directory |
| `--wandb` | Enable W&B logging |
| `--name` | Experiment name |
| `--seed` | Random seed |

### ECD-specific

| Argument | Description |
|----------|-------------|
| `--ecd_lr` | ECD learning rate (Δt̂) |
| `--ecd_eta` | Concentration parameter |
| `--ecd_F0` | Loss offset |
| `--ecd_nu` | Bounce amplitude |
| `--ecd_wd` | Weight decay |

### Disordered Attention

| Argument | Description |
|----------|-------------|
| `--mean_Q` | Mean of bQ distribution |
| `--std_Q` | Std of bQ distribution |
| `--disable_k_bias` | Only use bQ, not bK |

## Reproducibility

For exact reproduction of paper results:

1. Use the provided config files
2. Set `--seed` explicitly
3. Use the same PyTorch/CUDA versions
4. Use FineWeb-Edu 10B dataset

Note: Due to GPU non-determinism in cuDNN, exact loss values may vary slightly across runs, but qualitative conclusions should hold.

## Citation

If you use this code, please cite:

```bibtex
@article{symmetry_breaking_transformers,
  title={Symmetry Breaking in Transformer Optimization},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use gradient accumulation:
```bash
--batch_size 4 --grad_accum 2  # Effective batch = 8
```

### Data Loading Errors

Ensure data is prepared correctly:
```bash
ls data/fineweb10B/*.bin  # Should show train/val shards
```

### WandB Not Available

Training works without WandB - just omit `--wandb` flag.

### ECD Divergence

If ECD diverges, try:
1. Lower `ecd_lr` (e.g., 0.5 instead of 1.0)
2. Higher `F0` (e.g., 2 instead of -1)
3. Lower `eta` (e.g., 50 instead of 100)

## Contact

For questions or issues, please open a GitHub issue.
