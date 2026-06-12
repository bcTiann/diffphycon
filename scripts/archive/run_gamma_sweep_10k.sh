#!/bin/bash
# Sweep gamma (prior_beta) using our newly-trained FOPC_10k and FOPC_w_10k.
# Both models trained on the same 10k dataset (data/free_u_f_1e4_front_rear_quarter/),
# so this gives a controlled, internally-consistent gamma ablation.
set -e

# ===== checkpoint config (edit if testing mid-training) =====
JOINT_EXP=FOPC_10k
JOINT_MILESTONE=10
JOINT_INTERVAL=2500
JOINT_DIM=64
JOINT_DIM_MULS="1 2 4 8"

PRIOR_EXP=FOPC_w_10k
# auto-detect latest available milestone (handy when prior model is still training)
PRIOR_MILESTONE=$(ls trained_models/burgers_w/${PRIOR_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$PRIOR_MILESTONE" ] && { echo "ERROR: no checkpoints found for $PRIOR_EXP"; exit 1; }
PRIOR_INTERVAL=625
PRIOR_DIM=64
PRIOR_DIM_MULS="1 2 4 8"
# ============================================================

GAMMAS=(1.0 0.9 0.7 0.5 0.3)

echo "============================================================"
echo "Config: joint=$JOINT_EXP m${JOINT_MILESTONE} dim=$JOINT_DIM"
echo "        prior=$PRIOR_EXP m${PRIOR_MILESTONE} dim=$PRIOR_DIM"
echo "============================================================"

for G in "${GAMMAS[@]}"; do
    TAG="gamma${G//./}_10k"   # e.g. 0.5 -> gamma05_10k
    echo "============================================================"
    echo ">>> Running prior_beta=${G}   tag=${TAG}"
    echo "============================================================"
    python inference/inference_1d_burgers.py \
        --exp_id ${JOINT_EXP} \
        --checkpoint ${JOINT_MILESTONE} \
        --checkpoint_interval ${JOINT_INTERVAL} \
        --exp_id__model_w ${PRIOR_EXP} \
        --checkpoint__model_w ${PRIOR_MILESTONE} \
        --checkpoint_interval__model_w ${PRIOR_INTERVAL} \
        --dim__model_w ${PRIOR_DIM} \
        --dim_muls__model_w ${PRIOR_DIM_MULS} \
        --eval_two_models True \
        --prior_beta ${G} \
        --dataset free_u_f_1e4_front_rear_quarter \
        --partial_control front_rear_quarter \
        --partially_observed None \
        --train_on_partially_observed None \
        --set_unobserved_to_zero_during_sampling False \
        --is_condition_u0 True \
        --is_condition_uT True \
        --J_scheduler cosine \
        --dim ${JOINT_DIM} \
        --dim_muls ${JOINT_DIM_MULS} \
        --save_file burgers_results/full_obs_partial_ctr/result_${TAG}.yaml \
        --save_tag ${TAG} \
        --n_test_samples 8
done

echo "============================================================"
echo "All done. Trajectory files in outputs/trajectories/:"
ls -la outputs/trajectories/inference_trajectories_gamma*_10k.npz
