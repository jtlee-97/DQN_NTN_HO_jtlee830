#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

env MPLCONFIGDIR="$PWD/.codex_mplconfig" \
/opt/anaconda3/envs/dqn-ho/bin/python -u scripts/train_simple_dqn_handover.py \
  --config configs/rl_train_simple_dqn_handover.yaml \
  --output outputs/final_dqn_ho_run \
  --episodes 3000 \
  --eval-episodes 128 \
  --eval-interval 500 \
  --log-interval 100 \
  --rl-reward-scale 20.0 \
  --reward-transform scale \
  --scenario-mode fixed \
  --train-scenario-count 1 \
  --train-seed-offset 10002 \
  --val-scenario-count 128 \
  --test-seed-offset 100000 \
  --initial-eval
