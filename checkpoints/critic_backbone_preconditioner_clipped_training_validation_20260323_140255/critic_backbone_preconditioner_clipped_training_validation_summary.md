# Critic Backbone Preconditioner Clipped Training Validation

- completed_groups: baseline_critic_adam, critic_backbone_preconditioner_clip_head_adam
- skipped_optional_groups: critic_backbone_preconditioner_clip_ln_focus_head_adam
- policy_ratio_mode: hierarchical_actor_joint_reward_aligned_credit
- seed: 2025
- training_num_epochs: 6
- training_time_steps: 100
- update_epochs: 10

| group_name | best_epoch | best_reward | final_reward | reward_gap | overall_advantage_action_alignment | theta_advantage_alignment | route_advantage_alignment | value_explained_variance | prediction_target_corr | prediction_std_over_target_std | target_bucket_prediction_slope | joint_action_decision_agreement_ratio_under_reward_aligned | critic_backbone_grad_norm | critic_head_grad_norm | critic_loss | critic_backbone_preconditioner_cap | critic_backbone_preconditioner_active_threshold | focus_parameter_names | critic_head_untouched | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_critic_adam | 1 | -147.19946389522 | -155.03216068503139 | -7.8326967898113935 | 0.0825670361518859 | 0.0245269630104303 | 0.0466426908969879 | 0.9255541563034058 | 0.9628052115440368 | 1.000746488571167 | 0.93869111952756 | 0.5708333253860474 | 0.7878794192494851 | 0.8459726313341692 | 0.2112846999429166 | 0.0 | 1e-12 | all_critic_backbone_parameters | True | D:\MEC_PPO\checkpoints\critic_backbone_preconditioner_clipped_training_validation_20260323_140255\baseline_critic_adam_20260323_140301_970993 |
| critic_backbone_preconditioner_clip_head_adam | 1 | -147.19946389522 | -155.03216068503139 | -7.8326967898113935 | 0.0697383806109428 | 0.0455991737544536 | 0.0469082780182361 | 0.9335055947303772 | 0.9664491415023804 | 0.9436819553375244 | 0.8944948366628711 | 0.5708333253860474 | 0.935843317205839 | 0.8863597698629044 | 0.2193928792141378 | 5000.0 | 1e-12 | all_critic_backbone_parameters | True | D:\MEC_PPO\checkpoints\critic_backbone_preconditioner_clipped_training_validation_20260323_140255\critic_backbone_preconditioner_clip_head_adam_20260323_142529_945575 |