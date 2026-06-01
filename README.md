# Final DQN Handover Package

This folder is the restored DQN-only setup used before the optimization
benchmark additions.

## Included Comparison

- Unsafe A3: `0 dB offset`, `0 dB hysteresis`, `0 s TTT`
- Standard A3: `1 dB offset`, `0.5 dB hysteresis`, `0.4 s TTT`
- Proposed DQN: two-action DQN handover timing policy

Optimization benchmark scripts and MDP/Q-learning comparison outputs are not
included in this folder.

## Re-run

From this folder:

```bash
./run_final_dqn_ho.sh
```

Equivalent command:

```bash
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
```

## Main Outputs

- `outputs/final_dqn_ho_run/simple_dqn_training_reward.png`
- `outputs/final_dqn_ho_run/simple_dqn_rl_training_reward_only.png`
- `outputs/final_dqn_ho_run/simple_dqn_eval_bars.png`
- `outputs/final_dqn_ho_run/simple_dqn_ho_summary.md`
- `outputs/final_dqn_ho_run/simple_dqn_ho_eval.csv`
- `outputs/final_dqn_ho_run/best_dqn_model.pt`
- `outputs/final_dqn_ho_run/train_log.csv`
