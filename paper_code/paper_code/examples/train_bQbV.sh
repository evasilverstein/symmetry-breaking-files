#!/bin/bash
# Train with bQ + bV symmetry breaking using ECD
# This matches the bQ+bV run configuration from seed 83

python scripts/train.py \
    --model 124m \
    --ctx 512 \
    --vocab 50304 \
    --optimizer ecd \
    --ecd_lr 1.0 \
    --ecd_eta 100 \
    --ecd_F0 2.0 \
    --ecd_nu 0.0 \
    --ecd_consEn \
    --use_v_bias \
    --mean_V 0.5 \
    --std_V 0.05 \
    --batch_size 8 \
    --grad_accum 1 \
    --train_tokens 5e8 \
    --valid_every_updates 1000 \
    --use_bf16 \
    --seed 83 \
    --data_root ./data/finewebedu10B \
    --log_dir runs \
    --name bQbV-ecd-seed83 \
    --wandb
