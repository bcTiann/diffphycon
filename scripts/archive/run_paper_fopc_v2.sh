#!/bin/bash
# run_paper_fopc_v2.sh — paper-faithful FM training + eval + viz for 1D Burgers FO-PC
#
# v2 additions over v1:
#   - EMA (β=0.995, paper-standard)
#   - LR cosine annealing (paper Table 5)
#   - Boundary loss masking (paper D.4)
#   - Resume from intermediate ckpt (if exists)
#   - Joint: dim=128, mults=(1,2,4) — paper Table 5
#   - Prior: dim=32,  mults=(1,2,4,8) — paper Table 5
#   - Defaults: batch=16, steps=190k (strict paper) — override w/ env vars
#
# Pipeline:  train → loss plot → eval (γ-sweep, 1000 steps) → trajectory plot → compare
#
# Usage:
#   bash run_paper_fopc_v2.sh                      # full pipeline (train→eval→viz→compare)
#   STAGE=smoke   bash run_paper_fopc_v2.sh        # 500-step smoke (~2 min)
#   STAGE=train   bash run_paper_fopc_v2.sh        # only training (4 models)
#   STAGE=eval    bash run_paper_fopc_v2.sh        # only eval+viz+compare (needs ckpts)
#   BATCH=64 NUM_STEPS=50000 bash run_paper_fopc_v2.sh   # quick mode (~7 hr total)
#
# Run from repo root. tmux-friendly. Detach: Ctrl+B D.

set -e

DATASET=free_u_f_paper_fopc
CKPT_DIR=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2
RESULTS_DIR=/root/autodl-tmp/diffphycon/flow/results/paper_fopc_v2
mkdir -p $CKPT_DIR $RESULTS_DIR

# Default = strict paper-faithful (Table 5); override with env vars for quick mode
NUM_STEPS=${NUM_STEPS:-190000}    # paper Table 5
BATCH=${BATCH:-16}                # paper Table 5
STAGE=${STAGE:-all}
SKIP_OT=${SKIP_OT:-0}             # set 1 to skip OT-CFM (vanilla only, paper-complete)
SKIP_PRIOR=${SKIP_PRIOR:-0}       # set 1 to skip prior (γ=1 only, DiffPhyCon-lite)

# Common training args for ALL 4 models
COMMON_TRAIN_ARGS="\
    --num_steps $NUM_STEPS --batch_size $BATCH --lr 1e-4 \
    --ema_decay 0.995 --lr_scheduler cosine \
    --dataset $DATASET --device cuda \
    --ckpt_every 10000 --print_every 500"

# --------------------------------------------------------------- smoke
if [ "$STAGE" = "smoke" ]; then
  echo "########## SMOKE: 500 steps dim=32 (~2 min) ##########"
  python flow/burgers_fm_train.py \
      --variant vanilla --model joint \
      --dim 32 --dim_mults 1 2 4 \
      --num_steps 500 --batch_size 8 --lr 1e-4 \
      --dataset $DATASET --device cuda \
      --save_path $CKPT_DIR/smoke_vanilla_joint.pt \
      --ckpt_every 0 --print_every 50
  echo ">>> SMOKE OK."
  exit 0
fi

# --------------------------------------------------------------- training
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
  echo "########## JOINT 1/2: vanilla (dim=128, mults=1,2,4) ##########"
  python flow/burgers_fm_train.py \
      --variant vanilla --model joint \
      --dim 128 --dim_mults 1 2 4 \
      --save_path $CKPT_DIR/vanilla_joint.pt \
      $COMMON_TRAIN_ARGS

  if [ "$SKIP_OT" != "1" ]; then
    echo "########## JOINT 2/2: OT-CFM ##########"
    python flow/burgers_fm_train.py \
        --variant ot --model joint \
        --dim 128 --dim_mults 1 2 4 \
        --save_path $CKPT_DIR/ot_joint.pt \
        $COMMON_TRAIN_ARGS
  else
    echo "########## SKIPPED: OT joint (SKIP_OT=1) ##########"
  fi

  if [ "$SKIP_PRIOR" != "1" ]; then
    echo "########## PRIOR 1/2: vanilla (dim=32, mults=1,2,4,8) ##########"
    python flow/burgers_fm_train.py \
        --variant vanilla --model prior \
        --dim 32 --dim_mults 1 2 4 8 \
        --save_path $CKPT_DIR/vanilla_prior.pt \
        $COMMON_TRAIN_ARGS

    if [ "$SKIP_OT" != "1" ]; then
      echo "########## PRIOR 2/2: OT-CFM ##########"
      python flow/burgers_fm_train.py \
          --variant ot --model prior \
          --dim 32 --dim_mults 1 2 4 8 \
          --save_path $CKPT_DIR/ot_prior.pt \
          $COMMON_TRAIN_ARGS
    fi
  else
    echo "########## SKIPPED: priors (SKIP_PRIOR=1) ##########"
  fi
fi

# --------------------------------------------------------------- post-training: plots + eval
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
  echo "########## PLOT: loss curves ##########"
  python flow/plot_loss.py \
      --ckpt_dir $CKPT_DIR \
      --out $RESULTS_DIR/loss_curves.png

  # Build --variants list dynamically based on SKIP_OT
  EVAL_VARIANTS="vanilla"
  if [ "$SKIP_OT" != "1" ]; then EVAL_VARIANTS="vanilla ot"; fi

  echo "########## EVAL: γ-sweep, 1000 sampling steps, N_TEST=50 ##########"
  python flow/burgers_fm_eval_v2.py \
      --ckpt_dir $CKPT_DIR \
      --dataset $DATASET \
      --out_dir $RESULTS_DIR \
      --n_test 50 --n_steps 1000 \
      --gammas 0.0 0.3 0.5 0.7 1.0 \
      --variants $EVAL_VARIANTS

  echo "########## VIZ: 5 test trajectories (γ=0, γ=1) ##########"
  python flow/plot_trajectories.py \
      --ckpt_dir $CKPT_DIR \
      --dataset $DATASET \
      --out_dir $RESULTS_DIR \
      --n_samples 5 --n_steps 1000 \
      --gammas 0.0 1.0 \
      --variants $EVAL_VARIANTS

  echo "########## COMPARE: ours vs paper Table 1 + Table 25 ##########"
  python flow/compare_to_paper.py \
      --eval_csv $RESULTS_DIR/eval_table.csv \
      --out $RESULTS_DIR/comparison.md
fi

echo "########## ALL DONE ##########"
echo "Checkpoints:  $CKPT_DIR"
echo "Results:      $RESULTS_DIR"
ls -la $RESULTS_DIR
