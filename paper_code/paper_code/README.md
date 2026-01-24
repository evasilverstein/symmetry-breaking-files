# ECD Symmetry Breaking Paper Code

This package implements the ECD (Energy Conserving Descent) optimizer with symmetry-breaking mechanisms for transformer training, as described in the accompanying paper.

## Key Idea

Transformers have continuous rotational symmetries in attention (Q → RQ, K → RK leaves scores invariant). These symmetries create conserved Noether currents that constrain ECD optimization. Breaking these symmetries via random query biases (bQ) enables ECD to match/exceed Adam performance.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training with ECD + Symmetry Breaking (bQ + bV)

```bash
python scripts/train.py \
    --model 124m \
    --optimizer ecd \
    --ecd_lr 1.0 --ecd_eta 100 --ecd_F0 2.0 \
    --use_v_bias --mean_V 0.5 --std_V 0.05 \
    --train_tokens 5e8 \
    --data_root ./data/finewebedu10B \
    --wandb --name ecd-bQbV-seed42
```

### Symmetric Baseline (no bQ)

```bash
python scripts/train.py \
    --symmetric \
    --optimizer adam \
    --adam_lr 1e-4 \
    --train_tokens 5e8 \
    --data_root ./data/finewebedu10B
```

### Evaluation

```bash
# Logic puzzle evaluation
python scripts/eval_logic_puzzles.py \
    --checkpoint runs/ecd-bQbV-seed42/model_best.pt \
    --use_q_bias --no_k_bias --use_v_bias \
    --mean_Q 0.5 --mean_V 0.5 \
    --output results/logic_eval.json

# bQ alignment analysis
python scripts/analyze_bQ.py \
    --checkpoint runs/ecd-bQbV-seed42/model_best.pt \
    --output_prefix results/bQ_analysis \
    --data_path ./data/finewebedu10B/finewebedu_train_000099.bin
```

## Package Structure

```
paper_code/
├── ecd_symbreak/
│   ├── config.py          # GPTConfig, model presets
│   ├── optimizer.py       # ECD_q1_scaled optimizer
│   ├── utils.py           # Utilities
│   ├── model/
│   │   ├── attention.py   # CausalSelfAttention, FullSymmetryBrokenAttention
│   │   ├── mlp.py         # MLP, RandomPReLU1d, AsymmetricMLPPreLU
│   │   ├── block.py       # SymmetricBlock, DisorderedBlock
│   │   └── gpt.py         # GPT model
│   └── data/
│       └── loader.py      # FineWebShards data loader
├── scripts/
│   ├── train.py           # Unified training script
│   ├── eval_logic_puzzles.py  # Logic puzzle evaluation
│   └── analyze_bQ.py      # bQ alignment analysis
├── examples/
│   ├── train_symmetric.sh # Symmetric baseline example
│   ├── train_bQbV.sh      # bQ+bV symmetry breaking example
│   └── evaluate.sh        # Evaluation examples
└── requirements.txt
```

## Key Hyperparameter defaults, can be varied

### ECD Optimizer (defaults from tested runs)
- `ecd_lr`: 1.0 (learning rate, rescaled by 1/sqrt(eta) internally)
- `ecd_eta`: 100 (concentration parameter)
- `ecd_F0`: 2.0 (loss offset)
- `ecd_nu`: 0.0 (bounce amplitude)
- `ecd_consEn`: True (energy conservation)

### Symmetry Breaking Biases
- `mean_Q`: 0.5 (mean of bQ distribution)
- `mean_V`: 0.5 (mean of bV distribution, for V-O sector)
- `std_V`: 0.05 (std of bV distribution)

### Model
- `vocab_size`: 50304 (matches GPT-2 tokenizer)
- `context_length`: 512
- `model`: "124m" (12 layers, 12 heads, 768 dim)

## Architecture Modes

### Symmetric Mode (`--symmetric`)
- Standard causal self-attention (no bQ)
- Optional bV for V-O symmetry breaking
- PReLU or GeLU MLP:  PreLU was used in the main paper examples.

### Disordered Mode (default)
- bQ breaks O(d_k) rotational symmetry in Q-K sector
- bK intentionally omitted (cancels in softmax)
- Optional bV breaks O(d_v) symmetry in V-O sector
- PReLU MLP (learnable per-feature slopes)

## Theoretical Background

### Why Symmetry Breaking Matters for ECD

1. **Noether's Theorem**: Symmetries in the Hamiltonian create conserved quantities
2. **Conservation Constraints**: Angular momenta are preserved during ECD dynamics
3. **Optimization Problems**: Conservation restricts parameter space exploration
4. **Solution**: Random bQ breaks symmetry, eliminating constraints

### The bQ Mechanism

```python
q = WQ @ x + bQ  # Add random bias to queries
```

- bQ ~ N(mean_Q, std_Q) resampled each batch
- Non-learned (if learned, symmetry persists)
- Model learns to use ⟨bQ⟩ direction semantically


