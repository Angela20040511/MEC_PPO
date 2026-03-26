# Joint Training Critic Drift Trace

- classification: `A`
- detail: The first clear critic semantic drop appears immediately after a critic step.

## Initial Probe Metrics

```json
{
  "epoch": 0,
  "update_epoch": 0,
  "minibatch_id": 0,
  "trace_stage": "before_actor",
  "probe_pearson_value_vs_value_target": -0.0051825209520757,
  "probe_spearman_value_vs_value_target": -0.0381024740636348,
  "probe_pearson_value_vs_return": -0.0051825200207531,
  "probe_spearman_value_vs_return": -0.0381024740636348,
  "probe_pearson_value_vs_advantage": -0.0189545210450887,
  "probe_spearman_value_vs_advantage": -0.0463525801897048
}
```

## First Divergence

```json
{
  "where": "after_critic",
  "epoch": 0,
  "update_epoch": 1,
  "minibatch_id": 7
}
```

## Epoch Summary

| epoch | route_update_count | actor_loss | critic_loss | probe value-target pearson | probe return pearson |
|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 0.114249 | 2.320915 | 0.374490 | 0.374490 |
| 1 | 40 | -0.001851 | 0.949673 | 0.358563 | 0.358563 |