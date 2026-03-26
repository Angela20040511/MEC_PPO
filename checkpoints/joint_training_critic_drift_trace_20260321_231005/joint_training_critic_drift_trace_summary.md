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
  "probe_pearson_value_vs_value_target": 0.3278252184391022,
  "probe_spearman_value_vs_value_target": -0.0318063385784626,
  "probe_pearson_value_vs_return": 0.3278252184391022,
  "probe_spearman_value_vs_return": -0.0318063385784626,
  "probe_pearson_value_vs_advantage": -0.8455946445465088,
  "probe_spearman_value_vs_advantage": -0.7409899830818176
}
```

## First Divergence

```json
{
  "where": "after_critic",
  "epoch": 0,
  "update_epoch": 2,
  "minibatch_id": 9
}
```

## Epoch Summary

| epoch | route_update_count | actor_loss | critic_loss | probe value-target pearson | probe return pearson |
|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 0.001083 | 0.404729 | 0.384415 | 0.384415 |
| 1 | 40 | -0.166941 | 0.205976 | 0.697173 | 0.697173 |