#!/bin/bash
# FOPC gamma sweep — PAPER-FAITHFUL CONFIG
# Adds: (1) w_scheduler sigmoid_flip (per paper FOPC config)
#       (2) extends beta range to include >1 values (paper uses beta=1.5 for FOPC)
# Tag suffix: _FOPC_paper to distinguish from earlier sweep (no w_scheduler, only beta<=1).
set -e

JOINT_EXP=FOPC_10k
JOINT_MILESTONE=10
JOINT_INTERVAL=2500
JOINT_DIM=64
JOINT_DIM_MULS="1 2 4 8"

PRIOR_EXP=FOPC_w_10k
PRIOR_MILESTONE=$(ls trained_models/burgers_w/${PRIOR_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$PRIOR_MILESTONE" ] && { echo "ERROR: no checkpoints for $PRIOR_EXP"; exit 1; }
PRIOR_INTERVAL=625
PRIOR_DIM=64
PRIOR_DIM_MULS="1 2 4 8"

# 7 beta values: 5 below 1 (away from prior), 1 = baseline, 2 above 1 (paper FOPC's regime)
BETAS=(0.3 0.5 0.7 0.9 1.0 1.5 2.5)

echo "============================================================"
echo "FOPC gamma sweep (paper config)"
echo "  joint=$JOINT_EXP m${JOINT_MILESTONE}"
echo "  prior=$PRIOR_EXP m${PRIOR_MILESTONE}"
echo "  w_scheduler=sigmoid_flip (paper FOPC default)"
echo "============================================================"

for B in "${BETAS[@]}"; do
    TAG="beta${B//./}_FOPC_paper"      # e.g. 1.5 -> beta15_FOPC_paper
    echo "============================================================"
    echo ">>> Running prior_beta=${B}   tag=${TAG}"
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
        --prior_beta ${B} \
        --normalize_beta False \
        --w_scheduler sigmoid_flip \
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
echo "FOPC paper sweep done. Files in outputs/trajectories/:"
ls -la outputs/trajectories/inference_trajectories_beta*_FOPC_paper.npz
