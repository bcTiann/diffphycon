#!/bin/bash
# Train FOPC_w: the prior model p(f) for DiffPhyCon's prior reweighting.
# Pairs with the existing FOPC joint p(u, f) checkpoint to enable gamma experiments.
#
# Reduced training scale (20k steps vs paper's 200k) to fit Mac MPS budget.
# Architecture: dim=64 dim_muls=1,2,4,8 (script default, smaller than FOPC's 128).
# This is fine — FOPC and FOPC_w don't need matching arch.
python train/train_1d_burgers.py \
--is_condition_u0 True \
--is_condition_uT True \
--exp_id FOPC_w \
--dim 64 \
--dataset free_u_f_1e5_front_rear_quarter \
--partially_observed None \
--train_on_partially_observed None \
--dim_muls 1 2 4 8 \
--train_num_steps 20000 \
--checkpoint_interval 2000 \
--is_model_w True \
--expand_condition False
