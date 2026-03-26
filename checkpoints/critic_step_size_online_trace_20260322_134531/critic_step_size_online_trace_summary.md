# Critic Step Size Online Trace

- mode: `hierarchical_actor_joint_reward_aligned_credit`
- actor_frozen: `True`
- trace_epochs: `2`
- interpretation: Shrinking only the critic step size clearly suppresses the first after-optimizer semantic drop; critic step radius looks like the primary driver.

## Group Comparison

```json
{
  "current_step_size": {
    "first_divergence": {
      "epoch": 0,
      "update_epoch": 0,
      "minibatch_id": 0,
      "trace_stage": "after_optimizer_step",
      "delta_metrics": {
        "probe_pearson_value_vs_value_target": 0.007221132516860962,
        "probe_spearman_value_vs_value_target": -0.02735562063753605,
        "probe_pearson_value_vs_return": 0.00722116231918335,
        "probe_spearman_value_vs_return": -0.02735562063753605
      },
      "critic_param_delta_norm": 0.0530448050090163,
      "update_over_grad_ratio": 0.012232140352256848
    },
    "optimizer_probe_target_drop_min": -0.034036025404930115,
    "optimizer_probe_target_drop_mean": 0.0024344105273485183
  },
  "quarter_step_size": {
    "first_divergence": {
      "epoch": 0,
      "update_epoch": 0,
      "minibatch_id": 0,
      "trace_stage": "after_optimizer_step",
      "delta_metrics": {
        "probe_pearson_value_vs_value_target": 0.002223491668701172,
        "probe_spearman_value_vs_value_target": -0.010421188548207283,
        "probe_pearson_value_vs_return": 0.0022235214710235596,
        "probe_spearman_value_vs_return": -0.010421188548207283
      },
      "critic_param_delta_norm": 0.01326120067015765,
      "update_over_grad_ratio": 0.0030580349538326854
    },
    "optimizer_probe_target_drop_min": -0.0027863681316375732,
    "optimizer_probe_target_drop_mean": 0.001784234866499901
  },
  "tenth_step_size": {
    "first_divergence": null,
    "optimizer_probe_target_drop_min": -0.0006086230278015137,
    "optimizer_probe_target_drop_mean": 0.0011822652071714402
  },
  "relative_drop_reduction_vs_current": {
    "quarter_step_size_min_drop_ratio": 0.08186526183618278,
    "tenth_step_size_min_drop_ratio": 0.01788173033016231
  }
}
```