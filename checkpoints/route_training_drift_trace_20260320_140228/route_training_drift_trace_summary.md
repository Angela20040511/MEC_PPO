# Route Training Drift Trace

- classification: `完整训练动态塌缩`
- detail: 前两轮 trace 内没有出现统计口径失配，也没有出现单步 route step 被 guard 压掉；如果长程训练最后变成 route_update_count=0，更像是更晚发生的训练动态塌缩。

## Epoch Summary

| mode | epoch | train route_update_count | trace counted sum | trace optimizer sum | trace route_param_delta sum |
|---|---:|---:|---:|---:|---:|
| hierarchical_actor_true_conditional_route_policy | 0 | 40 | 40 | 40 | 1.887997 |
| hierarchical_actor_true_conditional_route_policy | 1 | 40 | 40 | 40 | 2.243472 |
| hierarchical_actor_true_conditional_route_candidate_score_credit | 0 | 0 | 0 | 0 | 0.000000 |
| hierarchical_actor_true_conditional_route_candidate_score_credit | 1 | 0 | 0 | 0 | 0.000000 |

## First Divergence

- none within traced epochs 0-1