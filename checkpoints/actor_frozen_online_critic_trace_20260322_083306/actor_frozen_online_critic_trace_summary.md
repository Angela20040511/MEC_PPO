# Actor Frozen Online Critic Trace

- classification: `A`
- detail: The first clear critic semantic drop appears immediately after a critic step.
- actor_param_delta_norm_max: `0.000000`

## First Divergence

```json
{
  "where": "after_critic",
  "epoch": 0,
  "update_epoch": 2,
  "minibatch_id": 9
}
```

## Comparison To Previous Normal Trace

```json
{
  "normal_trace_summary_path": "D:\\MEC_PPO\\checkpoints\\joint_training_critic_drift_trace_20260321_231005\\joint_training_critic_drift_trace_summary.json",
  "normal_trace_classification": "A",
  "normal_trace_first_divergence": {
    "where": "after_critic",
    "epoch": 0,
    "update_epoch": 2,
    "minibatch_id": 9
  },
  "normal_trace_probe_summary": {
    "initial_before_actor_probe_target_pearson": 0.3278252184391022,
    "final_after_critic_probe_target_pearson": 0.6733196973800659,
    "initial_before_actor_probe_return_pearson": 0.3278252184391022,
    "final_after_critic_probe_return_pearson": 0.6733196377754211
  },
  "frozen_trace_probe_summary": {
    "initial_before_actor_probe_target_pearson": 0.3278252184391022,
    "final_after_critic_probe_target_pearson": 0.6733486652374268,
    "initial_before_actor_probe_return_pearson": 0.3278252184391022,
    "final_after_critic_probe_return_pearson": 0.6733488440513611
  }
}
```