#!/bin/bash
# Train symmetric baseline with ECD
# This matches the symmetric run configuration from seed 83

python scripts/train.py \
    --model 124m \
    --ctx 512 \
    --vocab 50304 \
    --symmetric \
    --use_prelu \
    --optimizer ecd \
    --ecd_lr 1.0 \
    --ecd_eta 100 \
    --ecd_F0 2.0 \
    --ecd_nu 0.0 \
    --ecd_consEn \
    --batch_size 8 \
    --grad_accum 1 \
    --train_tokens 5e8 \
    --valid_every_updates 500 \
    --use_bf16 \
    --seed 83 \
    --data_root ./data/finewebedu10B \
    --log_dir runs \
    --name symmetric-ecd-seed83 \
    --wandb
