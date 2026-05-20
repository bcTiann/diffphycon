#!/bin/bash
# Train POPC_w prior p(f | u_0, u_T) under partial observation.
# Despite is_model_w=True zeroing u[1..T-1] internally, u_0 and u_T are still seen
# by the model as conditioning - and under partial observation those endpoints
# have their middle 1/2 masked. So POPC_w != FOPC_w and must be trained separately.
# Matched architecture/budget with FOPC_w_10k for fair comparison.
python train/train_1d_burgers.py \
--is_condition_u0 True \
--is_condition_uT True \
--exp_id POPC_w_10k \
--dim 64 \
--dataset free_u_f_1e4_front_rear_quarter \
--partially_observed front_rear_quarter \
--train_on_partially_observed front_rear_quarter \
--dim_muls 1 2 4 8 \
--train_num_steps 6250 \
--checkpoint_interval 625 \
--batch_size 64 \
--is_model_w True \
--expand_condition False \
"$@"
