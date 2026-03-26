# Critic Optimizer Rule Ablation

- checkpoint: `checkpoints\dense_policy_joint_reward_aligned_credit_experiment_20260321_170716\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_172521_442663\best_model.pt`
- target minibatch: epoch `0` / update_epoch `2` / minibatch `9`
- interpretation: Small SGD barely harms probe semantics, but current Adam is clearly worse; the optimizer rule is amplifying or distorting the bad direction.

## Group Comparison

```json
{
  "no_op": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.6659913659095764,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.35978466272354126,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.3597846031188965,
    "critic_param_delta_norm": 0.0,
    "critic_backbone_delta_norm": 0.0,
    "critic_head_delta_norm": 0.0,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.0
  },
  "small_sgd": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.6660298109054565,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.3597657382488251,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.3597657382488251,
    "critic_param_delta_norm": 4.9995618115901255e-05,
    "critic_backbone_delta_norm": 4.049642622100018e-05,
    "critic_head_delta_norm": 2.9290090402671225e-05,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 9.999127780609873e-05
  },
  "current_adam": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.6915565133094788,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.33826130628585815,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.33826130628585815,
    "critic_param_delta_norm": 0.06524962277310246,
    "critic_backbone_delta_norm": 0.06470263497995109,
    "critic_head_delta_norm": 0.003795852149678181,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.13049929980510297
  }
}
```