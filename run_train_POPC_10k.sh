#!/bin/bash
# Train POPC joint p(u, f) on 10k dataset under partial observation:
# u in middle 1/2 is masked to 0 during training (both data loading + loss).
# Architecture & training budget matched to FOPC_w_10k for fair comparison.
# Paper uses the same 200000 steps for both FOPC and POPC; we use 6250.
python train/train_1d_burgers.py \
--is_condition_u0 True \
--is_condition_uT True \
--exp_id POPC_10k \
--dim 64 \
--dataset free_u_f_1e4_front_rear_quarter \
--partially_observed front_rear_quarter \
--train_on_partially_observed front_rear_quarter \
--dim_muls 1 2 4 8 \
--train_num_steps 6250 \
--checkpoint_interval 625 \
--batch_size 64 \
--is_model_w False \
--expand_condition False \
"$@"
