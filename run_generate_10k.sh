#!/bin/bash
# Generate 10k-sample Burgers dataset (8k train + 2k test) for FOPC setting.
# Saves to data/free_u_f_1e4_front_rear_quarter/ (mirrors paper's _1e5_ naming convention).
# CPU is safest for one-shot dataset generation; PDE solver does batched ops on tensors.
python dataset/apps/generate_burgers.py \
--device cpu \
--train_samples 8000 \
--test_samples 2000 \
--partial_control front_rear_quarter \
--save_path free_u_f_1e4_front_rear_quarter/
