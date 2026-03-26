# Policy Surrogate Block Conclusion

## Experiment root
- checkpoints\dense_policy_ratio_block_experiment_20260317_181047

## Fixed mainline
- topology=dense
- reward_mode=per_sensor
- value_target_mode=popart_return_norm
- value_loss_mode=huber
- critic_arch_mode=stronger_critic_backbone
- critic_input_mode=normalized_augmented_critic_input
- actor_input_mode=normalized_augmented_actor_input
- actor_raw_input_mode=baseline_actor_raw
- seed=2025

## Real action block layout
- action_dim=72
- action_block_count=24
- each block has 3 dims: theta, logit_BS1, logit_BS2
- block 00: [0:3) S1/LatCrit -> theta[0], logit_BS1[1], logit_BS2[2]
- block 01: [3:6) S1/ComHvy -> theta[3], logit_BS1[4], logit_BS2[5]
- block 02: [6:9) S1/BWHvy -> theta[6], logit_BS1[7], logit_BS2[8]
- block 03: [9:12) S2/LatCrit -> theta[9], logit_BS1[10], logit_BS2[11]
- block 04: [12:15) S2/ComHvy -> theta[12], logit_BS1[13], logit_BS2[14]
- block 05: [15:18) S2/BWHvy -> theta[15], logit_BS1[16], logit_BS2[17]
- block 06: [18:21) S3/LatCrit -> theta[18], logit_BS1[19], logit_BS2[20]
- block 07: [21:24) S3/ComHvy -> theta[21], logit_BS1[22], logit_BS2[23]
- block 08: [24:27) S3/BWHvy -> theta[24], logit_BS1[25], logit_BS2[26]
- block 09: [27:30) S4/LatCrit -> theta[27], logit_BS1[28], logit_BS2[29]
- block 10: [30:33) S4/ComHvy -> theta[30], logit_BS1[31], logit_BS2[32]
- block 11: [33:36) S4/BWHvy -> theta[33], logit_BS1[34], logit_BS2[35]
- block 12: [36:39) S5/LatCrit -> theta[36], logit_BS1[37], logit_BS2[38]
- block 13: [39:42) S5/ComHvy -> theta[39], logit_BS1[40], logit_BS2[41]
- block 14: [42:45) S5/BWHvy -> theta[42], logit_BS1[43], logit_BS2[44]
- block 15: [45:48) S6/LatCrit -> theta[45], logit_BS1[46], logit_BS2[47]
- block 16: [48:51) S6/ComHvy -> theta[48], logit_BS1[49], logit_BS2[50]
- block 17: [51:54) S6/BWHvy -> theta[51], logit_BS1[52], logit_BS2[53]
- block 18: [54:57) S7/LatCrit -> theta[54], logit_BS1[55], logit_BS2[56]
- block 19: [57:60) S7/ComHvy -> theta[57], logit_BS1[58], logit_BS2[59]
- block 20: [60:63) S7/BWHvy -> theta[60], logit_BS1[61], logit_BS2[62]
- block 21: [63:66) S8/LatCrit -> theta[63], logit_BS1[64], logit_BS2[65]
- block 22: [66:69) S8/ComHvy -> theta[66], logit_BS1[67], logit_BS2[68]
- block 23: [69:72) S8/BWHvy -> theta[69], logit_BS1[70], logit_BS2[71]

## Mode definitions
- current_joint_sum_ratio: sum log-prob deltas over all 72 dims first, then build one global PPO ratio/surrogate.
- mean_logprob_ratio: average log-prob deltas over all 72 dims first, then build one global PPO ratio/surrogate.
- blockwise_surrogate_mean: for each 3-dim dispatch block, sum block log-prob deltas, build ratio_block, apply block-local PPO clip, then average surrogate_block across 24 blocks.
- block_mean_ratio previous reference: previous experiment's block scaling baseline, included only as a historical reference.

## Key findings
- blockwise_surrogate_mean is meaningfully closer to action semantics than block_mean_ratio, because it changes the surrogate construction itself instead of only shrinking a global ratio by block count.
- blockwise_surrogate_mean partially alleviates clipping saturation versus current_joint_sum_ratio: final clip_fraction drops from 0.99 to 0.782917, positive_adv_clip_fraction from 1.0 to 0.906863, negative_adv_clip_fraction from 0.988095 to 0.75753.
- blockwise_surrogate_mean does not collapse into mean_logprob_ratio behavior: final ratio_max is 10.1494 versus 1.35599 for mean_logprob_ratio, and positive-advantage bucket selected-action gain remains materially larger.
- blockwise_surrogate_mean still does not solve the best_epoch==1 problem. All three current modes still have best_epoch=1.
- reward-wise, blockwise_surrogate_mean improves clearly over mean_logprob_ratio, but does not beat current_joint_sum_ratio on final_reward.
- critic health remains intact under blockwise_surrogate_mean: value_explained_variance=0.970731, prediction_target_corr=0.985422, prediction_std_over_target_std=0.967363, target_bucket_prediction_slope=0.954324.
- if the next step is needed, priority should move to how advantage enters the surrogate, especially whether block-specific weighting or blockwise advantages are needed, rather than returning to generic PPO hyperparameter tuning.

## Key comparison file
- checkpoints\dense_policy_ratio_block_experiment_20260317_181047\policy_surrogate_key_comparison.csv
- checkpoints\dense_policy_ratio_block_experiment_20260317_181047\policy_surrogate_key_comparison.json
- checkpoints\dense_policy_ratio_block_experiment_20260317_181047\policy_surrogate_key_comparison.md
