#!/bin/bash

# Stop any existing training process
pkill -f train_mps.py || true

# Run the training with optimized settings
python train_mps.py \
  --batch_size 256 \
  --d_model 64 \
  --nhead 4 \
  --num_encoder_layers 1 \
  --num_main_layers 1 \
  --use_subset 0.25 \
  --warmup_steps 200 \
  --lr 0.002 \
  --use_ema \
  --fast_mode \
  --val_subset 0.2 \
  --eval_every 2

echo "Training started with optimized settings for faster performance."
echo "Monitor the output to see the improved training speed." 