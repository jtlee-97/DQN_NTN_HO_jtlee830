# Final Proposed DQN Full Run Analysis

- Output: `outputs/final_dqn_ho_run`
- Scenario: 3 overlapping satellites x 19 beams = 57 cells.
- Interference: target cell plus 18 nearest non-target interferer cells.
- UE: starts at (0, 0), random heading, 1000 km/h.
- Simulation time: 18.0 s.
- RLF thresholds: Qout/Qin = -8.0/-6.0 dB.

| Metric | Unsafe A3 | Standard A3 | Proposed DQN | DQN vs Unsafe |
|---|---:|---:|---:|---:|
| Reward | 21.838258 | 16.572273 | 28.436122 | 6.598 |
| Avg SINR [dB] | 2.639254 | 2.094186 | 2.408029 | -0.231 dB |
| Outage time [s] | 1.601562 | 1.603125 | 1.548437 | 0.053 s saved |
| RLF count | 1.296875 | 1.890625 | 1.218750 | 0.078 fewer (6.0%) |
| UHO count | 2.570312 | 1.179688 | 0.257812 | 2.312 fewer (90.0%) |
| HO count | 6.265625 | 5.000000 | 6.257812 | 0.008 fewer (0.1%) |
| RB/s/UE | 3.565538 | 2.945747 | 2.848958 | -0.717 |
| Avg ToS [s] | 2.319980 | 2.172500 | 2.696838 | 0.377 s |

## Interpretation

- Proposed DQN produces the highest reward while keeping SINR in a healthy positive range.
- The main gain is mobility stability: UHO is strongly reduced while RLF and outage are slightly improved.
- SINR is slightly lower than Unsafe A3 because the policy avoids aggressively chasing every instantaneous strongest cell and instead favors stable time-of-stay.
