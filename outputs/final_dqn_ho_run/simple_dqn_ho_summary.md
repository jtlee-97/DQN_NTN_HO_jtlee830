# Proposed DQN Handover Evaluation

| Metric | Unsafe A3 | Standard A3 | Proposed DQN |
|---|---:|---:|---:|
| Reward | 21.838258 | 16.572273 | 28.436122 |
| Avg SINR [dB] | 2.639254 | 2.094186 | 2.408029 |
| Outage time [s] | 1.601562 | 1.603125 | 1.548437 |
| RLF count | 1.296875 | 1.890625 | 1.218750 |
| UHO count | 2.570312 | 1.179688 | 0.257812 |
| HO count | 6.265625 | 5.000000 | 6.257812 |
| RB/s/UE | 3.565538 | 2.945747 | 2.848958 |
| Avg ToS [s] | 2.319980 | 2.172500 | 2.696838 |

## Paired DQN - Unsafe A3

- unsafe_reward_gain_mean: 6.597864
- unsafe_sinr_gain_db_mean: -0.231225
- unsafe_outage_time_saved_s_mean: 0.053125
- unsafe_rlf_delta_mean: -0.078125
- unsafe_uho_delta_mean: -2.312500
- unsafe_extra_ho_mean: -0.007812
- safety_score: 23.416664
