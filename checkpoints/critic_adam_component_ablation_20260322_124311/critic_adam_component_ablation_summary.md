# Critic Adam Component Ablation

- checkpoint: `checkpoints\dense_policy_joint_reward_aligned_credit_experiment_20260321_170716\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_172521_442663\best_model.pt`
- target minibatch: epoch `0` / update_epoch `2` / minibatch `9`
- interpretation: Both norm-matched SGD and current Adam hurt probe semantics, but Adam still hurts more; update magnitude matters, with Adam rule adding extra damage.

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
    "probe_advantage_pearson_before": -0.46308663487434387,
    "probe_advantage_pearson_after": -0.46308663487434387,
    "critic_param_delta_norm": 0.0,
    "critic_backbone_delta_norm": 0.0,
    "critic_head_delta_norm": 0.0,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.0,
    "cosine_update_vs_grad": 0.0,
    "cosine_update_vs_neg_grad": 0.0
  },
  "current_adam": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.6915565133094788,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.33826130628585815,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.33826130628585815,
    "probe_advantage_pearson_before": -0.46308663487434387,
    "probe_advantage_pearson_after": -0.42099079489707947,
    "critic_param_delta_norm": 0.06524962277310246,
    "critic_backbone_delta_norm": 0.06470263497995109,
    "critic_head_delta_norm": 0.003795852149678181,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.13049929980510297,
    "cosine_update_vs_grad": -0.17548954958377713,
    "cosine_update_vs_neg_grad": 0.17548954958377713
  },
  "adam_fresh_state": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.7423577308654785,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.3132598400115967,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.31325986981391907,
    "probe_advantage_pearson_before": -0.46308663487434387,
    "probe_advantage_pearson_after": -0.5394993424415588,
    "critic_param_delta_norm": 0.14721167915333483,
    "critic_backbone_delta_norm": 0.13899312016861778,
    "critic_head_delta_norm": 0.00567891118532941,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.294423480721837,
    "cosine_update_vs_grad": -0.47870181908986426,
    "cosine_update_vs_neg_grad": 0.47870181908986426
  },
  "norm_matched_sgd": {
    "batch_target_pearson_before": 0.6659913659095764,
    "batch_target_pearson_after": 0.7364886403083801,
    "probe_target_pearson_before": 0.35978466272354126,
    "probe_target_pearson_after": 0.32462573051452637,
    "probe_return_pearson_before": 0.3597846031188965,
    "probe_return_pearson_after": 0.32462576031684875,
    "probe_advantage_pearson_before": -0.46308663487434387,
    "probe_advantage_pearson_after": -0.5119498372077942,
    "critic_param_delta_norm": 0.06524962607084558,
    "critic_backbone_delta_norm": 0.05284496682161811,
    "critic_head_delta_norm": 0.0382369996808741,
    "grad_norm": 0.4999997921103863,
    "update_over_grad_ratio": 0.13049930640059196,
    "cosine_update_vs_grad": -1.0000040768916578,
    "cosine_update_vs_neg_grad": 1.0000040768916578
  },
  "norm_matched_sgd_lr": 0.13049929980510297
}
```