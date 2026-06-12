#!/bin/bash
# Sweep gamma (prior_beta) at FOPC setting, saving each run with a distinct tag.
# Expected story for FOPC: gamma=1.0 == baseline; smaller gamma -> worse J and Energy.
set -e

GAMMAS=(1.0 0.9 0.7 0.5 0.3)

for G in "${GAMMAS[@]}"; do
    TAG="gamma${G//./}"   # e.g. 0.5 -> gamma05
    echo "============================================================"
    echo ">>> Running prior_beta=${G}   tag=${TAG}"
    echo "============================================================"
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
        --prior_beta ${G} \
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
        --save_file burgers_results/full_obs_partial_ctr/result_gamma${TAG}.yaml \
        --save_tag ${TAG} \
        --n_test_samples 8
done
echo "============================================================"
echo "All done. Trajectory files saved as inference_trajectories_gammaXX.npz"
ls -la inference_trajectories_gamma*.npz
