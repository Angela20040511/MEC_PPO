# Critic Backbone Preconditioner Detail Audit

- checkpoint: `checkpoints\dense_policy_joint_reward_aligned_credit_experiment_20260321_170716\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_172521_442663\best_model.pt`
- captured minibatch: epoch `0` / update_epoch `2` / minibatch `8`
- backbone heavier tail: `True`
- dominant module: `critic_backbone`
- interpretation: Use this audit to decide whether the next formal validation should constrain critic_backbone preconditioning geometry rather than replace the full optimizer.

## Module Summary

```json
{
  "all_critic": {
    "numel": 81921,
    "raw_grad_norm": 2.1316978931427,
    "m_hat_norm": 0.4448276460170746,
    "v_hat_norm": 0.022851908579468727,
    "preconditioned_grad_norm": 1199.8896484375,
    "full_adam_update_norm": 176.5117645263672,
    "actual_param_delta_norm": 0.0882558673620224,
    "update_over_grad_ratio": 0.041401676872659166,
    "actual_delta_norm_share": 0.9999999428510155,
    "cosine_raw_vs_momentum": 0.8772846596566559,
    "cosine_raw_vs_preconditioned": 0.5767704594153136,
    "cosine_raw_vs_full_adam": 0.44071987300844495,
    "cosine_momentum_vs_full_adam": 0.37673346279209896,
    "cosine_preconditioned_vs_full_adam": 0.7884127496015468,
    "cosine_full_adam_vs_actual_delta": -1.000000522404142,
    "preconditioner_mean": 5147649.0,
    "preconditioner_std": 21779436.0,
    "preconditioner_min": 9.854516983032227,
    "preconditioner_max": 100000000.0,
    "preconditioner_p50": 1895.2076416015625,
    "preconditioner_p90": 567210.875,
    "preconditioner_p95": 22738786.0,
    "preconditioner_p99": 100000000.0,
    "preconditioner_p999": 100000000.0
  },
  "critic_backbone": {
    "numel": 81792,
    "raw_grad_norm": 1.5452789068222046,
    "m_hat_norm": 0.19749508798122406,
    "v_hat_norm": 0.0006465954938903451,
    "preconditioned_grad_norm": 1199.3326416015625,
    "full_adam_update_norm": 176.2361602783203,
    "actual_param_delta_norm": 0.0881180614233017,
    "update_over_grad_ratio": 0.05702404985551279,
    "actual_delta_norm_share": 0.9984385063713312,
    "cosine_raw_vs_momentum": 0.8125154626515269,
    "cosine_raw_vs_preconditioned": 0.767690691312599,
    "cosine_raw_vs_full_adam": 0.5568672040187082,
    "cosine_momentum_vs_full_adam": 0.7392988094547476,
    "cosine_preconditioned_vs_full_adam": 0.7883127724085002,
    "cosine_full_adam_vs_actual_delta": -1.0000006078507446,
    "preconditioner_mean": 5155767.0,
    "preconditioner_std": 21795644.0,
    "preconditioner_min": 103.58946990966797,
    "preconditioner_max": 100000000.0,
    "preconditioner_p50": 1898.890625,
    "preconditioner_p90": 567912.25,
    "preconditioner_p95": 100000000.0,
    "preconditioner_p99": 100000000.0,
    "preconditioner_p999": 100000000.0
  },
  "critic_head": {
    "numel": 129,
    "raw_grad_norm": 1.4684171676635742,
    "m_hat_norm": 0.3985816538333893,
    "v_hat_norm": 0.022842759266495705,
    "preconditioned_grad_norm": 36.55624771118164,
    "full_adam_update_norm": 9.860003471374512,
    "actual_param_delta_norm": 0.004930002149194479,
    "update_over_grad_ratio": 0.0033573580163453803,
    "actual_delta_norm_share": 0.055860329911292715,
    "cosine_raw_vs_momentum": 0.997647473229859,
    "cosine_raw_vs_preconditioned": 0.9780407454766398,
    "cosine_raw_vs_full_adam": 0.9790780771963202,
    "cosine_momentum_vs_full_adam": 0.9791896102701065,
    "cosine_preconditioned_vs_full_adam": 0.9966918367791155,
    "cosine_full_adam_vs_actual_delta": -0.9999999914570783,
    "preconditioner_mean": 26.70696449279785,
    "preconditioner_std": 9.504945755004883,
    "preconditioner_min": 9.854516983032227,
    "preconditioner_max": 79.02842712402344,
    "preconditioner_p50": 24.523374557495117,
    "preconditioner_p90": 33.0206413269043,
    "preconditioner_p95": 38.908599853515625,
    "preconditioner_p99": 68.2275390625,
    "preconditioner_p999": 77.72119140625
  },
  "critic_block_value_head": {
    "numel": 0,
    "raw_grad_norm": 0.0,
    "m_hat_norm": 0.0,
    "v_hat_norm": 0.0,
    "preconditioned_grad_norm": 0.0,
    "full_adam_update_norm": 0.0,
    "actual_param_delta_norm": 0.0,
    "update_over_grad_ratio": 0.0,
    "actual_delta_norm_share": 0.0,
    "cosine_raw_vs_momentum": 0.0,
    "cosine_raw_vs_preconditioned": 0.0,
    "cosine_raw_vs_full_adam": 0.0,
    "cosine_momentum_vs_full_adam": 0.0,
    "cosine_preconditioned_vs_full_adam": 0.0,
    "cosine_full_adam_vs_actual_delta": 0.0,
    "preconditioner_mean": 0.0,
    "preconditioner_std": 0.0,
    "preconditioner_min": 0.0,
    "preconditioner_max": 0.0,
    "preconditioner_p50": 0.0,
    "preconditioner_p90": 0.0,
    "preconditioner_p95": 0.0,
    "preconditioner_p99": 0.0,
    "preconditioner_p999": 0.0
  },
  "critic_block_path_value_head": {
    "numel": 0,
    "raw_grad_norm": 0.0,
    "m_hat_norm": 0.0,
    "v_hat_norm": 0.0,
    "preconditioned_grad_norm": 0.0,
    "full_adam_update_norm": 0.0,
    "actual_param_delta_norm": 0.0,
    "update_over_grad_ratio": 0.0,
    "actual_delta_norm_share": 0.0,
    "cosine_raw_vs_momentum": 0.0,
    "cosine_raw_vs_preconditioned": 0.0,
    "cosine_raw_vs_full_adam": 0.0,
    "cosine_momentum_vs_full_adam": 0.0,
    "cosine_preconditioned_vs_full_adam": 0.0,
    "cosine_full_adam_vs_actual_delta": 0.0,
    "preconditioner_mean": 0.0,
    "preconditioner_std": 0.0,
    "preconditioner_min": 0.0,
    "preconditioner_max": 0.0,
    "preconditioner_p50": 0.0,
    "preconditioner_p90": 0.0,
    "preconditioner_p95": 0.0,
    "preconditioner_p99": 0.0,
    "preconditioner_p999": 0.0
  }
}
```

## Actual Step Effects

```json
{
  "current_batch": {
    "before_critic_loss": 0.3996136486530304,
    "after_critic_loss": 0.2771938443183899,
    "delta_critic_loss": -0.1224198043346405,
    "before_target_pearson": 0.6262403130531311,
    "after_target_pearson": 0.7379237413406372,
    "delta_target_pearson": 0.1116834282875061
  },
  "held_out_batch": {
    "before_critic_loss": 0.46690577268600464,
    "after_critic_loss": 0.4543265402317047,
    "delta_critic_loss": -0.012579232454299927,
    "before_target_pearson": 0.6415639519691467,
    "after_target_pearson": 0.6588694453239441,
    "delta_target_pearson": 0.017305493354797363
  },
  "probe": {
    "before_critic_loss": 0.9337701797485352,
    "after_critic_loss": 0.961993932723999,
    "delta_critic_loss": 0.028223752975463867,
    "before_target_pearson": 0.3689935803413391,
    "after_target_pearson": 0.3503134250640869,
    "delta_target_pearson": -0.018680155277252197
  }
}
```

## Top Layers

```json
[
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.0",
    "tensor_name": "critic_backbone.0.weight",
    "actual_delta_norm_share": 0.8088592997102075,
    "actual_param_delta_norm": 0.07138658314943314,
    "preconditioner_p99": 100000000.0,
    "preconditioner_p999": 100000000.0,
    "preconditioner_max": 100000000.0,
    "cosine_raw_vs_preconditioned": 0.8016259160896819,
    "cosine_raw_vs_full_adam": 0.555991951427621
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.3",
    "tensor_name": "critic_backbone.3.weight",
    "actual_delta_norm_share": 0.43749114253128946,
    "actual_param_delta_norm": 0.03861116245388985,
    "preconditioner_p99": 5106.79541015625,
    "preconditioner_p999": 6131.083984375,
    "preconditioner_max": 6699.0556640625,
    "cosine_raw_vs_preconditioned": 0.8868066589872036,
    "cosine_raw_vs_full_adam": 0.6867124565038497
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.6",
    "tensor_name": "critic_backbone.6.weight",
    "actual_delta_norm_share": 0.3497970802083281,
    "actual_param_delta_norm": 0.030871646478772163,
    "preconditioner_p99": 8108.86865234375,
    "preconditioner_p999": 10677.8798828125,
    "preconditioner_max": 13583.794921875,
    "cosine_raw_vs_preconditioned": 0.9458952100476392,
    "cosine_raw_vs_full_adam": 0.8035742665768637
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.7",
    "tensor_name": "critic_backbone.7.weight",
    "actual_delta_norm_share": 0.07408477284016221,
    "actual_param_delta_norm": 0.006538416258990765,
    "preconditioner_p99": 32326.91015625,
    "preconditioner_p999": 38283.25390625,
    "preconditioner_max": 38864.28515625,
    "cosine_raw_vs_preconditioned": 0.9582515415018178,
    "cosine_raw_vs_full_adam": 0.963738026307268
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.1",
    "tensor_name": "critic_backbone.1.bias",
    "actual_delta_norm_share": 0.06774038188863386,
    "actual_param_delta_norm": 0.005978486500680447,
    "preconditioner_p99": 3883.177734375,
    "preconditioner_p999": 4481.33837890625,
    "preconditioner_max": 4566.43310546875,
    "cosine_raw_vs_preconditioned": 0.8707784819773323,
    "cosine_raw_vs_full_adam": 0.8363664113724673
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.7",
    "tensor_name": "critic_backbone.7.bias",
    "actual_delta_norm_share": 0.06432609558918703,
    "actual_param_delta_norm": 0.005677155684679747,
    "preconditioner_p99": 3973.685302734375,
    "preconditioner_p999": 4331.8564453125,
    "preconditioner_max": 4353.17333984375,
    "cosine_raw_vs_preconditioned": 0.9670076453335804,
    "cosine_raw_vs_full_adam": 0.9630244827473607
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.0",
    "tensor_name": "critic_backbone.0.bias",
    "actual_delta_norm_share": 0.06389534139165304,
    "actual_param_delta_norm": 0.005639139097183943,
    "preconditioner_p99": 2103.200439453125,
    "preconditioner_p999": 2182.926025390625,
    "preconditioner_max": 2192.517578125,
    "cosine_raw_vs_preconditioned": 0.8687183347598956,
    "cosine_raw_vs_full_adam": 0.836666527387927
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.3",
    "tensor_name": "critic_backbone.3.bias",
    "actual_delta_norm_share": 0.0571623588686624,
    "actual_param_delta_norm": 0.005044913850724697,
    "preconditioner_p99": 2040.9979248046875,
    "preconditioner_p999": 2416.839111328125,
    "preconditioner_max": 2464.331787109375,
    "cosine_raw_vs_preconditioned": 0.8843818913956172,
    "cosine_raw_vs_full_adam": 0.8758683710224491
  },
  {
    "module_group": "critic_head",
    "layer_name": "critic_head",
    "tensor_name": "critic_head.weight",
    "actual_delta_norm_share": 0.055781006557474415,
    "actual_param_delta_norm": 0.0049230013974010944,
    "preconditioner_p99": 68.24854278564453,
    "preconditioner_p999": 77.73139953613281,
    "preconditioner_max": 79.02842712402344,
    "cosine_raw_vs_preconditioned": 0.981198049227661,
    "cosine_raw_vs_full_adam": 0.9822603483449434
  },
  {
    "module_group": "critic_backbone",
    "layer_name": "critic_backbone.4",
    "tensor_name": "critic_backbone.4.bias",
    "actual_delta_norm_share": 0.0537942398797058,
    "actual_param_delta_norm": 0.00474765757098794,
    "preconditioner_p99": 6268.85546875,
    "preconditioner_p999": 6962.2421875,
    "preconditioner_max": 7035.740234375,
    "cosine_raw_vs_preconditioned": 0.890537089357359,
    "cosine_raw_vs_full_adam": 0.8783558225508769
  }
]
```