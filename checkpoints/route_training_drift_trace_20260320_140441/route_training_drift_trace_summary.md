# Route Training Drift Trace

- classification: `完整训练控制路径未接入`
- detail: candidate 组在前两轮里没有生成任何 route-only trace row；这不是统计口径丢记，而是完整训练路径在进入 route branch trace 之前就已经被 mode 分发挡住了。

## Epoch Summary

| mode | epoch | train route_update_count | trace counted sum | trace optimizer sum | trace route_param_delta sum |
|---|---:|---:|---:|---:|---:|
| hierarchical_actor_true_conditional_route_policy | 0 | 40 | 40 | 40 | 1.887997 |
| hierarchical_actor_true_conditional_route_policy | 1 | 40 | 40 | 40 | 2.243472 |
| hierarchical_actor_true_conditional_route_candidate_score_credit | 0 | 0 | 0 | 0 | 0.000000 |
| hierarchical_actor_true_conditional_route_candidate_score_credit | 1 | 0 | 0 | 0 | 0.000000 |

## First Divergence

```json
{
  "where": "_uses_blockwise_policy_surrogate gate before the first route minibatch",
  "epoch": 0,
  "update_epoch": 0,
  "minibatch_id": 0
}
```