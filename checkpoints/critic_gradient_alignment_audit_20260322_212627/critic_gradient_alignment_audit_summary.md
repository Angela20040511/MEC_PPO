# Critic Gradient Alignment Audit

- checkpoint: `checkpoints\dense_policy_joint_reward_aligned_credit_experiment_20260321_170716\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_172521_442663\best_model.pt`
- captured minibatch: epoch `0` / update_epoch `2` / minibatch `8`
- target update norm: `5e-05`
- interpretation: Current-batch gradient direction appears misaligned with held-out/global semantics if its cosine to held-out/global is low or negative and its matched small-step update improves current_batch while worsening held_out_batch/probe more than heldout/blended/global directions.

## Gradient Cosines

```json
{
  "current_vs_heldout_cosine": 0.6008114180406094,
  "current_vs_global_cosine": 0.8179006510940898,
  "current_vs_blended_cosine": 0.8998173965551086,
  "heldout_vs_global_cosine": 0.9503147388466269,
  "current_vs_heldout_backbone_cosine": 0.24137743911888668,
  "current_vs_heldout_head_cosine": 0.989132827052515,
  "grad_norm_current": 2.131699207039702,
  "grad_norm_heldout": 2.034149548211698,
  "grad_norm_global": 1.882236064373162,
  "grad_norm_blended": 1.8636237215606646
}
```

## Direction Effects

```json
{
  "current_grad_small_step": {
    "current_batch": {
      "critic_loss_delta": -0.00021311640739440918,
      "target_pearson_delta": 6.22868537902832e-05,
      "return_pearson_delta": 6.22868537902832e-05,
      "advantage_pearson_delta": 2.63899564743042e-05
    },
    "held_out_batch": {
      "critic_loss_delta": -0.00012224912643432617,
      "target_pearson_delta": 6.854534149169922e-06,
      "return_pearson_delta": 7.033348083496094e-06,
      "advantage_pearson_delta": 7.778406143188477e-06
    },
    "probe": {
      "critic_loss_delta": -0.0001869797706604004,
      "target_pearson_delta": -1.2308359146118164e-05,
      "return_pearson_delta": -1.239776611328125e-05,
      "advantage_pearson_delta": -1.049041748046875e-05
    },
    "global_batch": {
      "critic_loss_delta": -0.0001539289951324463,
      "target_pearson_delta": 1.6987323760986328e-05,
      "return_pearson_delta": 1.6987323760986328e-05,
      "advantage_pearson_delta": 1.920759677886963e-05
    },
    "delta_metrics": {
      "critic_param_delta_norm": 5.0010856697186014e-05,
      "critic_backbone_delta_norm": 3.6245430602640285e-05,
      "critic_head_delta_norm": 3.4458011376393474e-05,
      "update_over_grad_ratio": 2.346057424837835e-05,
      "scale_factor": 2.345548126722894e-05
    }
  },
  "heldout_grad_small_step": {
    "current_batch": {
      "critic_loss_delta": -0.00012806057929992676,
      "target_pearson_delta": -1.71661376953125e-05,
      "return_pearson_delta": -1.722574234008789e-05,
      "advantage_pearson_delta": -1.9840896129608154e-05
    },
    "held_out_batch": {
      "critic_loss_delta": -0.0002034604549407959,
      "target_pearson_delta": 1.6987323760986328e-05,
      "return_pearson_delta": 1.710653305053711e-05,
      "advantage_pearson_delta": 1.341104507446289e-05
    },
    "probe": {
      "critic_loss_delta": -0.00024300813674926758,
      "target_pearson_delta": -1.665949821472168e-05,
      "return_pearson_delta": -1.6748905181884766e-05,
      "advantage_pearson_delta": -2.866983413696289e-05
    },
    "global_batch": {
      "critic_loss_delta": -0.00017887353897094727,
      "target_pearson_delta": 1.0192394256591797e-05,
      "return_pearson_delta": 1.0073184967041016e-05,
      "advantage_pearson_delta": 6.407499313354492e-07
    },
    "delta_metrics": {
      "critic_param_delta_norm": 5.000636478686546e-05,
      "critic_backbone_delta_norm": 3.581259252007129e-05,
      "critic_head_delta_norm": 3.490121396439372e-05,
      "update_over_grad_ratio": 2.458342763900651e-05,
      "scale_factor": 2.458029867176381e-05
    }
  },
  "global_grad_small_step": {
    "current_batch": {
      "critic_loss_delta": -0.00017431378364562988,
      "target_pearson_delta": 1.0311603546142578e-05,
      "return_pearson_delta": 1.0251998901367188e-05,
      "advantage_pearson_delta": -5.37186861038208e-06
    },
    "held_out_batch": {
      "critic_loss_delta": -0.0001933276653289795,
      "target_pearson_delta": 1.430511474609375e-05,
      "return_pearson_delta": 1.430511474609375e-05,
      "advantage_pearson_delta": 1.1861324310302734e-05
    },
    "probe": {
      "critic_loss_delta": -0.0002524256706237793,
      "target_pearson_delta": -1.633167266845703e-05,
      "return_pearson_delta": -1.6361474990844727e-05,
      "advantage_pearson_delta": -2.390146255493164e-05
    },
    "global_batch": {
      "critic_loss_delta": -0.0001882016658782959,
      "target_pearson_delta": 1.2993812561035156e-05,
      "return_pearson_delta": 1.2874603271484375e-05,
      "advantage_pearson_delta": 6.750226020812988e-06
    },
    "delta_metrics": {
      "critic_param_delta_norm": 5.000712395395527e-05,
      "critic_backbone_delta_norm": 3.084270277378902e-05,
      "critic_head_delta_norm": 3.936292839403535e-05,
      "update_over_grad_ratio": 2.6567946857358753e-05,
      "scale_factor": 2.6564162020016932e-05
    }
  },
  "blended_grad_small_step": {
    "current_batch": {
      "critic_loss_delta": -0.00019165873527526855,
      "target_pearson_delta": 2.6226043701171875e-05,
      "return_pearson_delta": 2.6285648345947266e-05,
      "advantage_pearson_delta": 4.328787326812744e-06
    },
    "held_out_batch": {
      "critic_loss_delta": -0.00018084049224853516,
      "target_pearson_delta": 1.329183578491211e-05,
      "return_pearson_delta": 1.33514404296875e-05,
      "advantage_pearson_delta": 1.1771917343139648e-05
    },
    "probe": {
      "critic_loss_delta": -0.00023955106735229492,
      "target_pearson_delta": -1.6242265701293945e-05,
      "return_pearson_delta": -1.6242265701293945e-05,
      "advantage_pearson_delta": -2.1696090698242188e-05
    },
    "global_batch": {
      "critic_loss_delta": -0.00018554925918579102,
      "target_pearson_delta": 1.52587890625e-05,
      "return_pearson_delta": 1.52587890625e-05,
      "advantage_pearson_delta": 1.1339783668518066e-05
    },
    "delta_metrics": {
      "critic_param_delta_norm": 4.9987508485739525e-05,
      "critic_backbone_delta_norm": 3.173826686200645e-05,
      "critic_head_delta_norm": 3.8619081050795914e-05,
      "update_over_grad_ratio": 2.682275753622226e-05,
      "scale_factor": 2.682946034795301e-05
    }
  }
}
```