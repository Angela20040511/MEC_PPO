# Critic Batch Global Loss Trace

- mode: `hierarchical_actor_joint_reward_aligned_credit`
- actor_frozen: `True`
- heldout_sample_count: `64`
- interpretation: Optimizer steps repeatedly improve current-batch fit while hurting held-out and probe semantics.

## First Batch-Better / Global-Worse Point

```json
{
  "epoch": 0,
  "update_epoch": 2,
  "minibatch_id": 8,
  "trace_stage": "after_optimizer_step",
  "current_batch": {
    "delta_critic_loss": -0.06531363725662231,
    "delta_target_pearson": 0.044605374336242676,
    "delta_return_pearson": 0.044605374336242676,
    "delta_advantage_pearson": 0.08738052845001221
  },
  "held_out_batch": {
    "delta_critic_loss": 0.0018028616905212402,
    "delta_target_pearson": 0.03636890649795532,
    "delta_return_pearson": 0.03636893630027771,
    "delta_advantage_pearson": 0.11512525286525488
  },
  "probe": {
    "delta_critic_loss": 0.010218322277069092,
    "delta_target_pearson": -0.02780255675315857,
    "delta_return_pearson": -0.02780255675315857,
    "delta_advantage_pearson": 0.045938640832901
  }
}
```

## After Optimizer Delta Summary

```json
{
  "current_batch": {
    "mean_delta_critic_loss_after_optimizer_step": -0.048742698214482516,
    "mean_delta_target_pearson_after_optimizer_step": 0.05219768187962472,
    "mean_delta_return_pearson_after_optimizer_step": 0.05219767433591187,
    "mean_delta_advantage_pearson_after_optimizer_step": 0.05326473837485537,
    "min_delta_target_pearson_after_optimizer_step": -0.06940281391143799,
    "max_delta_target_pearson_after_optimizer_step": 0.8543315529823303
  },
  "held_out_batch": {
    "mean_delta_critic_loss_after_optimizer_step": -0.005875662714242935,
    "mean_delta_target_pearson_after_optimizer_step": 0.0005867978557944298,
    "mean_delta_return_pearson_after_optimizer_step": 0.0005867982283234597,
    "mean_delta_advantage_pearson_after_optimizer_step": 0.003955932473763824,
    "min_delta_target_pearson_after_optimizer_step": -0.047076284885406494,
    "max_delta_target_pearson_after_optimizer_step": 0.06951473653316498
  },
  "probe": {
    "mean_delta_critic_loss_after_optimizer_step": -0.007110663875937462,
    "mean_delta_target_pearson_after_optimizer_step": 0.002817161753773689,
    "mean_delta_return_pearson_after_optimizer_step": 0.002817162126302719,
    "mean_delta_advantage_pearson_after_optimizer_step": 0.008029226679354906,
    "min_delta_target_pearson_after_optimizer_step": -0.04078546166419983,
    "max_delta_target_pearson_after_optimizer_step": 0.04474470019340515
  }
}
```