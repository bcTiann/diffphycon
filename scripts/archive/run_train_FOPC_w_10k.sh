#!/bin/bash
# Train FOPC_w prior p(f) on the new 10k dataset.
# Same arch as FOPC_10k for controlled comparison.
python train/train_1d_burgers.py \
--is_condition_u0 True \
--is_condition_uT True \
--exp_id FOPC_w_10k \
--dim 64 \
--dataset free_u_f_1e4_front_rear_quarter \
--partially_observed None \
--train_on_partially_observed None \
--dim_muls 1 2 4 8 \
--train_num_steps 6250 \
--checkpoint_interval 625 \
--batch_size 64 \
--is_model_w True \
--expand_condition False \
"$@"
