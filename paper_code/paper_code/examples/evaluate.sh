#!/bin/bash
# Evaluation examples for trained models

# Logic puzzle evaluation for bQ+bV model
echo "=== Evaluating bQ+bV model on logic puzzles ==="
python scripts/eval_logic_puzzles.py \
    --checkpoint runs/bQbV-ecd-seed83/model_best.pt \
    --use_q_bias \
    --no_k_bias \
    --use_v_bias \
    --mean_Q 0.5 \
    --mean_V 0.5 \
    --output results/logic_eval_bQbV.json

# Logic puzzle evaluation for symmetric model
echo "=== Evaluating symmetric model on logic puzzles ==="
python scripts/eval_logic_puzzles.py \
    --checkpoint runs/symmetric-ecd-seed83/model_best.pt \
    --no_q_bias \
    --no_k_bias \
    --no_v_bias \
    --output results/logic_eval_symmetric.json

# bQ alignment analysis
echo "=== Running bQ alignment analysis ==="
python scripts/analyze_bQ.py \
    --checkpoint runs/bQbV-ecd-seed83/model_best.pt \
    --output_prefix results/bQ_analysis \
    --data_path ./data/finewebedu10B/finewebedu_train_000099.bin \
    --num_batches 100 \
    --top_k 15 \
    --batch_size 32

echo "=== Evaluation complete ==="
echo "Results saved to results/"
