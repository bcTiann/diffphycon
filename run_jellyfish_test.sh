#!/bin/bash
# Quick jellyfish inference test on MPS (50 timesteps for speed)
export PYTHONUNBUFFERED=1
export PYTHONPATH=/Users/baochen/diffphycon
python -u inference/inference_2d_jellyfish.py \
    --num_batches 1 \
    --batch_size 1 \
    --w_prob_exp 1.0 \
    --sampling_timesteps 50
