#!/bin/bash
# Option B: keep both hard conditions, but push heavy energy regularization (wfs=0.5).
# Compare against:
#   - experiment 1 (both hard, no guidance):   J_actual = 0.00116, Energy = 1989
# Question: how much energy can we save before J_actual breaks?
python inference/inference_1d_burgers.py \
--exp_id FOPC \
--checkpoint 170 \
--checkpoint_interval 1000 \
--dataset free_u_f_1e5_front_rear_quarter \
--partial_control front_rear_quarter \
--partially_observed None \
--train_on_partially_observed None \
--set_unobserved_to_zero_during_sampling False \
--is_condition_u0 True \
--is_condition_uT True \
--J_scheduler cosine \
--wus 0 \
--wfs 0.5 \
--dim 128 \
--dim_muls 1 2 4 \
--save_file burgers_results/full_obs_partial_ctr/result_lite.yaml \
--n_test_samples 8
