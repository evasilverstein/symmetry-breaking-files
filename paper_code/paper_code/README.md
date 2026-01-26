# Code accompanying paper: Symmetry Breaking in Transformers for Efficient and Interpretable Training

This package implements the ECD (Energy Conserving Descent) optimizer and an explicit symmetry-breaking mechanism using unlearned stochastic attention biases for transformer training, as described in the accompanying paper.  It also includes code to analyze two additional measures beyond validation loss:  (1)  downstream performance on a suite of simple logic puzzles and (2) an analysis of alignment of token classes' key vectors with the preferred bias directions.  Code is available for all versions of symmetry breaking in via attention biases and both types of MLP blocks (with PreLU taken first in the paper, with GeLU included as an ablation subsection in the paper with corresponding code in the gelu_ablation directory folder here).  

## Key Idea

Transformers have continuous rotational symmetries in attention (Q → RQ, K → RK leaves scores invariant and similarly for the value-output sector), leading to many redundant parameters carried along in the computation. These symmetries create conserved Noether currents and can limit chaotic exploration, impeding Energy Conserving Descent (ECD) optimization. Breaking these symmetries via random query and value biases (bQ) enables ECD to compete with Adam and SOAP.  

Moreover, singling out preferred directions introduces a new learning opportunity for any optimizer:  the model  can learn to align or anti-align the key vectors of particular, semantically interpretable, token classes with the query bias direction, amplifying or suppressing their attention.  The paper explores this effect and its role as a predictor of performance improvements on downstream reasoning.  



## Installation

```bash
pip install -r requirements.txt
```

### Included Optimizers

This package includes four optimizers:
- **ECD** (`ecd_symbreak/optimizer.py`) - Energy Conserving Descent q=1 version 
- **Adam** - PyTorch AdamW - has intrinsic symmetry breaking via coordinate axes
- **SGDM** - SGD with momentum
- **SOAP** (`soap.py`) - Shampoo-like optimizer, with preconditioning to restore some symmetry

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

### Symmetric Baseline (no bQ or bV)

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

## Training Output and Logging

Each training run saves to `runs/<run-name>/` with:
- `model_best.pt` - Checkpoint with best validation loss
- `model_final.pt` - Final checkpoint
- `training_log.csv` - Training curves (update, train_loss, val_loss, consumed_tokens, tok_per_s, elapsed_sec)

To plot training curves without W&B:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/my-run/training_log.csv")
plt.plot(df["consumed_tokens"], df["val_loss"])
plt.xlabel("Tokens"); plt.ylabel("Validation Loss")
plt.savefig("training_curve.png")
```

## Package Structure

```
  paper_code/                                                                                                                                                
  ├── soap.py               # SOAP optimizer (included)                                                                                                      
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
  ├── gelu_ablation/         # GELU MLP ablation study (NOT main paper results)                                                                              
  │   ├── README.md                                                                                                                                          
  │   ├── code/                                                                                                                                              
  │   │   └── train_gpt_gelu_ablation.py                                                                                                                     
  │   ├── scripts/                                                                                                                                           
  │   │   ├── eval_logic_puzzles_gelu.py                                                                                                                     
  │   │   └── analyze_bQ_gelu.py                                                                                                                             
  │   └── analysis_results/                                                                                                                                  
  │       ├── GELU_ABLATION_SUMMARY.md                                                                                                                       
  │       ├── logic_puzzles/*.json                                                                                                                           
  │       └── bQ_alignment/*.png, *.txt                                                                                                                      
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
- `mean_V`: 0.5 (mean of bV distribution, for V-O sector), 0.0 also tested
- `std_V`: 0.05 (std of bV distribution)

### Model
- `vocab_size`: 50304 (matches GPT-2 tokenizer)
- `context_length`: 512
- `model`: "124m" (12 layers, 12 heads, 768 dim)

## Architecture Modes

### Symmetric Mode (`--symmetric`)
- Standard causal self-attention (no bQ)
- Optional bV for V-O symmetry breaking
- PReLU or GeLU MLP:  PreLU was used in the main paper examples, GELU can be found in a special section and gelu_ablation code folder.

### Disordered Mode (default)
- bQ breaks O(d_k) rotational symmetry in Q-K sector
- bK intentionally omitted (cancels in softmax)
- Optional bV breaks O(d_v) symmetry in V-O sector
- PReLU MLP (learnable per-feature slopes); GELU version can be found in gelu_ablation folder




