#!/bin/bash
# Option A FIXED: sigmoid_flip schedule (large at t->0, small at t->T) + boosted wus
# Replaces the broken cosine schedule that killed guidance in 99% of sampling.
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
--is_condition_uT False \
--J_scheduler sigmoid_flip \
--wus 100.0 \
--wfs 0 \
--dim 128 \
--dim_muls 1 2 4 \
--save_file burgers_results/full_obs_partial_ctr/result_lite.yaml \
--n_test_samples 8
