# Critic Step Internal Trace

- classification: `B`
- detail: The first clear semantic drop appears only after critic optimizer.step().
- actor_param_delta_norm_max: `0.000000`

## First Divergence

```json
{
  "where": "after_optimizer_step",
  "from_stage": "after_backward_before_step",
  "epoch": 0,
  "update_epoch": 2,
  "minibatch_id": 9,
  "trace_scope": "minibatch",
  "delta_metrics": {
    "probe_pearson_value_vs_value_target": -0.016391456127166748,
    "probe_spearman_value_vs_value_target": 0.06578376889228821,
    "probe_pearson_value_vs_return": -0.016391456127166748,
    "probe_spearman_value_vs_return": 0.06578376889228821,
    "probe_pearson_value_vs_advantage": 0.04879772663116455,
    "probe_spearman_value_vs_advantage": 0.09129396080970764
  }
}
```

## Comparison To Previous Frozen-Actor Trace

```json
{
  "actor_frozen_trace_summary_path": "checkpoints\\actor_frozen_online_critic_trace_20260322_083306\\actor_frozen_online_critic_trace_summary.json",
  "actor_frozen_classification": "A",
  "actor_frozen_first_divergence": {
    "where": "after_critic",
    "epoch": 0,
    "update_epoch": 2,
    "minibatch_id": 9
  }
}
```