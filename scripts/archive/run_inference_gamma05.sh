#!/bin/bash
# Test DiffPhyCon's prior reweighting at gamma=0.5 (prior_beta=0.5).
# Uses partially-trained FOPC_w (milestone 6 = step 12000) as the prior model.
# Pairs with the existing FOPC joint checkpoint (milestone 170).
python inference/inference_1d_burgers.py \
--exp_id FOPC \
--checkpoint 170 \
--checkpoint_interval 1000 \
--exp_id__model_w FOPC_w \
--checkpoint__model_w 10 \
--checkpoint_interval__model_w 2000 \
--dim__model_w 64 \
--dim_muls__model_w 1 2 4 8 \
--eval_two_models True \
--prior_beta 0.5 \
--dataset free_u_f_1e5_front_rear_quarter \
--partial_control front_rear_quarter \
--partially_observed None \
--train_on_partially_observed None \
--set_unobserved_to_zero_during_sampling False \
--is_condition_u0 True \
--is_condition_uT True \
--J_scheduler cosine \
--dim 128 \
--dim_muls 1 2 4 \
--save_file burgers_results/full_obs_partial_ctr/result_gamma05.yaml \
--n_test_samples 8
