#!/bin/bash
# Example training commands for the symmetry breaking experiments

# Make sure you're in the repository root
cd "$(dirname "$0")/.."

# 1. Prepare data first (only needed once)
echo "Step 1: Prepare FineWeb data (if not already done)"
echo "  cd data && python fineweb.py -v 10B"
echo ""

# 2. ECD with Disordered Attention (symmetry breaking)
echo "Step 2: Train ECD with disordered attention"
echo "python train_gpt_disorder.py \\"
echo "    --model 124m \\"
echo "    --optimizer ecd \\"
echo "    --ecd_lr 1.0 \\"
echo "    --ecd_eta 100 \\"
echo "    --ecd_F0 2 \\"
echo "    --train_tokens 500000000 \\"
echo "    --data_root data/fineweb10B \\"
echo "    --seed 42 \\"
echo "    --name ecd-disordered-seed42"
echo ""

# 3. ECD with Symmetric Attention (baseline)
echo "Step 3: Train ECD with symmetric attention (baseline)"
echo "python train_gpt_symmetric.py \\"
echo "    --model 124m \\"
echo "    --optimizer ecd \\"
echo "    --ecd_lr 1.0 \\"
echo "    --ecd_eta 100 \\"
echo "    --ecd_F0 2 \\"
echo "    --train_tokens 500000000 \\"
echo "    --data_root data/fineweb10B \\"
echo "    --seed 42 \\"
echo "    --name ecd-symmetric-seed42"
echo ""

# 4. AdamW with Disordered Attention
echo "Step 4: Train AdamW with disordered attention"
echo "python train_gpt_disorder.py \\"
echo "    --model 124m \\"
echo "    --optimizer adam \\"
echo "    --adam_lr 0.0001 \\"
echo "    --adam_wd 0.01 \\"
echo "    --train_tokens 500000000 \\"
echo "    --data_root data/fineweb10B \\"
echo "    --seed 42 \\"
echo "    --name adamw-disordered-seed42"
echo ""

# 5. AdamW with Symmetric Attention
echo "Step 5: Train AdamW with symmetric attention"
echo "python train_gpt_symmetric.py \\"
echo "    --model 124m \\"
echo "    --optimizer adam \\"
echo "    --adam_lr 0.0001 \\"
echo "    --adam_wd 0.01 \\"
echo "    --train_tokens 500000000 \\"
echo "    --data_root data/fineweb10B \\"
echo "    --seed 42 \\"
echo "    --name adamw-symmetric-seed42"
echo ""

echo "================================================="
echo "After training, run bQ alignment analysis:"
echo "python analysis/analysis_bQ_batch.py \\"
echo "    --checkpoint path/to/checkpoint.pt \\"
echo "    --output_prefix results/analysis"
echo "================================================="
