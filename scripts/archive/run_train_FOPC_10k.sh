#!/bin/bash
# Train FOPC joint p(u,f) on the new 10k dataset.
# Same architecture as FOPC_w (dim=64) for controlled comparison with flow matching later.
# 25k steps should converge based on FOPC_w's trajectory (loss flattened by ~12k).
python train/train_1d_burgers.py \
--is_condition_u0 True \
--is_condition_uT True \
--exp_id FOPC_10k \
--dim 64 \
--dataset free_u_f_1e4_front_rear_quarter \
--partially_observed None \
--train_on_partially_observed None \
--dim_muls 1 2 4 8 \
--train_num_steps 25000 \
--checkpoint_interval 2500 \
--batch_size 64 \
--is_model_w False \
--expand_condition False \
"$@"
