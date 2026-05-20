#!/bin/bash
# POPC gamma sweep — PAPER-FAITHFUL CONFIG
# Paper POPC uses beta=0.9 + w_scheduler sigmoid_flip (per scripts/burgers_inference_partial_obs_partial_ctr.sh)
# This sweep covers both sides of beta=1 to map the full Pareto curve.
set -e

JOINT_EXP=POPC_10k
JOINT_INTERVAL=625
JOINT_MILESTONE=$(ls trained_models/burgers/${JOINT_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$JOINT_MILESTONE" ] && { echo "ERROR: no checkpoints for $JOINT_EXP"; exit 1; }
JOINT_DIM=64
JOINT_DIM_MULS="1 2 4 8"

PRIOR_EXP=POPC_w_10k
PRIOR_INTERVAL=625
PRIOR_MILESTONE=$(ls trained_models/burgers_w/${PRIOR_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$PRIOR_MILESTONE" ] && { echo "ERROR: no checkpoints for $PRIOR_EXP"; exit 1; }
PRIOR_DIM=64
PRIOR_DIM_MULS="1 2 4 8"

BETAS=(0.3 0.5 0.7 0.9 1.0 1.5 2.5)

echo "============================================================"
echo "POPC gamma sweep (paper config)"
echo "  joint=$JOINT_EXP m${JOINT_MILESTONE}"
echo "  prior=$PRIOR_EXP m${PRIOR_MILESTONE}"
echo "  w_scheduler=sigmoid_flip (paper POPC default)"
echo "============================================================"

for B in "${BETAS[@]}"; do
    TAG="beta${B//./}_POPC_paper"
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
        --partially_observed front_rear_quarter \
        --train_on_partially_observed front_rear_quarter \
        --set_unobserved_to_zero_during_sampling False \
        --is_condition_u0 True \
        --is_condition_uT True \
        --J_scheduler cosine \
        --dim ${JOINT_DIM} \
        --dim_muls ${JOINT_DIM_MULS} \
        --save_file burgers_results/partial_obs_partial_ctr/result_${TAG}.yaml \
        --save_tag ${TAG} \
        --n_test_samples 8
done

echo "============================================================"
echo "POPC paper sweep done. Files in outputs/trajectories/:"
ls -la outputs/trajectories/inference_trajectories_beta*_POPC_paper.npz
