#!/bin/bash
# Gamma sweep for POPC (partial observation, partial control).
# Joint model: POPC_10k (newly trained with partial-obs masking).
# Prior model: REUSES FOPC_w_10k (because p(f) doesn't depend on observation level).
# Expected: gamma < 1 helps here (unlike FOPC).
set -e

# ===== checkpoint config =====
JOINT_EXP=POPC_10k
JOINT_INTERVAL=625
# auto-detect latest available milestone
JOINT_MILESTONE=$(ls trained_models/burgers/${JOINT_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$JOINT_MILESTONE" ] && { echo "ERROR: no checkpoints for $JOINT_EXP"; exit 1; }
JOINT_DIM=64
JOINT_DIM_MULS="1 2 4 8"

PRIOR_EXP=POPC_w_10k    # POPC-specific prior (sees partially-observed u_0, u_T)
PRIOR_INTERVAL=625
PRIOR_MILESTONE=$(ls trained_models/burgers_w/${PRIOR_EXP}/cos10000-model-*.pt 2>/dev/null \
    | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -z "$PRIOR_MILESTONE" ] && { echo "ERROR: no checkpoints for $PRIOR_EXP -- did you train it?"; exit 1; }
PRIOR_DIM=64
PRIOR_DIM_MULS="1 2 4 8"
# ============================================================

GAMMAS=(1.0 0.9 0.7 0.5 0.3)

echo "============================================================"
echo "POPC gamma sweep"
echo "  joint=$JOINT_EXP m${JOINT_MILESTONE} dim=$JOINT_DIM"
echo "  prior=$PRIOR_EXP m${PRIOR_MILESTONE} dim=$PRIOR_DIM"
echo "============================================================"

for G in "${GAMMAS[@]}"; do
    TAG="gamma${G//./}_POPC_10k"
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
echo "POPC gamma sweep done. Files in outputs/trajectories/:"
ls -la outputs/trajectories/inference_trajectories_gamma*_POPC_10k.npz
