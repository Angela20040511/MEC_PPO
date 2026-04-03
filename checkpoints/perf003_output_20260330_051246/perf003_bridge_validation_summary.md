# Bridge Raw Rows
| mode_name | seed | group_name | device | best_epoch | best_reward | final_reward | value_explained_variance | prediction_target_corr | critic_loss_current_batch | critic_loss_heldout_batch | would_reject_rate | backbone_active_preconditioner_p99 | run_dir | wall_clock_sec | vector_env_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compatible_default | 2024 | 005_pure_blended_baseline_adam | cuda | 2 | -145.8959771216727 | -156.3600768867247 | 0.982740581035614 | 0.9918352365493774 | 0.1142183391668368 | 0.0492577161872759 | 0.7 | 5324739.0 | D:\MEC_PPO\checkpoints\perf003_output_20260330_051246\bridge_runs\perf003_compatible_default_005_pure_blended_baseline_adam_20260330_060237_358852 | 1249.749156299993 | 1 |
| compatible_default | 2024 | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam | cuda | 2 | -144.8648308856695 | -156.46292276363965 | 0.9862428307533264 | 0.9935135841369628 | 0.1104580821469426 | 0.0436883960152044 | 0.65 | 4686898.0 | D:\MEC_PPO\checkpoints\perf003_output_20260330_051246\bridge_runs\perf003_compatible_default_005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_20260330_062328_351561 | 1232.2147355000052 | 1 |
| vectorized_2env_aligned | 2024 | 005_pure_blended_baseline_adam | cuda | 2 | -148.310148078634 | -160.10763708244588 | 0.9977622628211976 | 0.9989063143730164 | 0.0637894235168849 | 0.0115366089182706 | 0.7142857142857143 | 6945305.144999818 | D:\MEC_PPO\checkpoints\perf003_output_20260330_051246\bridge_runs\perf003_vectorized_2env_aligned_005_pure_blended_baseline_adam_20260330_064401_712921 | 2453.993875 | 2 |
| vectorized_2env_aligned | 2024 | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam | cuda | 2 | -148.1802423066127 | -161.2782268588265 | 0.994496524333954 | 0.997246265411377 | 0.0679834423753033 | 0.0115912225962217 | 0.6285714285714286 | 5800355.5 | D:\MEC_PPO\checkpoints\perf003_output_20260330_051246\bridge_runs\perf003_vectorized_2env_aligned_005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_20260330_072457_570854 | 2467.1386700000003 | 2 |

# Bridge Comparison Rows
| mode_name | comparison | metric_name | delta |
| --- | --- | --- | --- |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | best_reward | 1.0311462360031953 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | final_reward | -0.10284587691495517 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | value_explained_variance | 0.0035022497177124023 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | prediction_target_corr | 0.0016783475875853382 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_current_batch | -0.0037602570198942004 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_heldout_batch | -0.0055693201720715055 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | would_reject_rate | -0.04999999999999993 |
| compatible_default | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | backbone_active_preconditioner_p99 | -637841.0 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | best_reward | 0.1299057720212886 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | final_reward | -1.1705897763806377 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | value_explained_variance | -0.0032657384872436523 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | prediction_target_corr | -0.0016600489616394043 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_current_batch | 0.0041940188584184 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_heldout_batch | 5.4613677951100004e-05 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | would_reject_rate | -0.08571428571428574 |
| vectorized_2env_aligned | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | backbone_active_preconditioner_p99 | -1144949.644999818 |