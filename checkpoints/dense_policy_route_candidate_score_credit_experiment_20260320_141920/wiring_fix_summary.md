# Wiring Fix Summary

- first divergence before fix: `_uses_blockwise_policy_surrogate` gate before the first route minibatch
- root cause: `hierarchical_actor_true_conditional_route_candidate_score_credit` was missing from `_uses_blockwise_policy_surrogate()`
- fix applied: added candidate mode to that whitelist so full training now enters split-advantage construction and alternating route updates

## Trace Self-Check

| mode | epoch | trace_rows | optimizer_step_executed_sum | counted_into_route_update_count_sum |
|---|---:|---:|---:|---:|
| candidate before fix | 0 | 0 | 0 | 0 |
| candidate before fix | 1 | 0 | 0 | 0 |
| candidate after fix | 0 | 40 | 40 | 40 |
| candidate after fix | 1 | 40 | 40 | 40 |

## Rewired A/B

| metric | baseline | candidate rewired |
|---|---:|---:|
| best_epoch | 1 | 1 |
| best_reward | -148.6247 | -148.8283 |
| final_reward | -163.4585 | -288.8703 |
| reward_gap | -14.8338 | -140.0420 |
| overall_advantage_action_alignment | -0.0780 | -0.1091 |
| theta_advantage_alignment | 0.4447 | 0.4363 |
| route_advantage_alignment | 0.4958 | -0.0590 |
| route_approx_kl | 0.0191 | -0.0240 |
| route_clip_fraction | 0.2384 | 0.2180 |
| route_update_count | 40 | 40 |
| route_head_grad_norm | 0.1704 | 0.0761 |
| route_backbone_grad_norm | 0.1756 | 0.1297 |
| value_explained_variance | 0.9331 | 0.8455 |
| prediction_target_corr | 0.9679 | 0.9248 |
| prediction_std_over_target_std | 0.9070 | 0.8256 |
| target_bucket_prediction_slope | 0.8740 | 0.7371 |
