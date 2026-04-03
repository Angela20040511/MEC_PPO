# Bridge Raw Rows
| mode_name | seed | group_name | device | best_epoch | best_reward | final_reward | value_explained_variance | prediction_target_corr | critic_loss_current_batch | critic_loss_heldout_batch | would_reject_rate | backbone_active_preconditioner_p99 | run_dir | wall_clock_sec | vector_env_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compatible_default | 2024 | 005_pure_blended_baseline_adam | cuda | 2 | -145.8959771216727 | -156.3600768867247 | 0.982740581035614 | 0.9918352365493774 | 0.1142183391668368 | 0.0492577161872759 | 0.7 | 5324739.0 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_compatible_default_005_pure_blended_baseline_adam_20260329_223439_596149 | 1054.885299200003 | 1 |
| compatible_default | 2024 | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam | cuda | 2 | -144.8648308856695 | -156.46292276363965 | 0.9862428307533264 | 0.9935135841369628 | 0.1104580821469426 | 0.0436883960152044 | 0.65 | 4686898.0 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_compatible_default_005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_20260329_225215_535156 | 943.462947299995 | 1 |
| vectorized_2env | 2024 | 005_pure_blended_baseline_adam | cuda | 4 | -144.10652604246536 | -148.82326575533398 | 0.9448822736740112 | 0.973235249519348 | 0.0759065344871487 | 0.026338629622478 | 0.85 | 5710250.395000052 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_vectorized_2env_005_pure_blended_baseline_adam_20260329_230759_970122 | 627.7763373000053 | 2 |
| vectorized_2env | 2024 | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam | cuda | 4 | -144.60565090305306 | -149.5515028754458 | 0.9084007143974304 | 0.9535791277885436 | 0.0763421007082797 | 0.0225057595293037 | 0.7 | 4969789.5 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_vectorized_2env_005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_20260329_231828_859320 | 622.6984167999908 | 2 |
| vectorized_4env | 2024 | 005_pure_blended_baseline_adam | cuda | 3 | -143.49183612799763 | -149.4081177375328 | 0.8897405862808228 | 0.9432605504989624 | 0.0784009052309556 | 0.0202507323934696 | 0.7 | 5198166.0 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_vectorized_4env_005_pure_blended_baseline_adam_20260329_232852_662373 | 446.790462799996 | 4 |
| vectorized_4env | 2024 | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam | cuda | 3 | -144.19332451615912 | -151.5710636882511 | 0.9277086853981018 | 0.963283121585846 | 0.0656742293504066 | 0.0229169304016977 | 0.8 | 4648649.0 | D:\MEC_PPO\checkpoints\perf002_output_20260329_215104\bridge_runs\perf002_vectorized_4env_005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_20260329_233620_444905 | 447.737861600006 | 4 |

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
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | best_reward | -0.499124860587699 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | final_reward | -0.7282371201118281 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | value_explained_variance | -0.03648155927658081 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | prediction_target_corr | -0.019656121730804443 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_current_batch | 0.00043556622113100074 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_heldout_batch | -0.0038328700931743002 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | would_reject_rate | -0.15000000000000002 |
| vectorized_2env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | backbone_active_preconditioner_p99 | -740460.8950000517 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | best_reward | -0.7014883881614935 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | final_reward | -2.1629459507182958 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | value_explained_variance | 0.03796809911727905 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | prediction_target_corr | 0.020022571086883545 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_current_batch | -0.012726675880549002 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | critic_loss_heldout_batch | 0.0026661980082281003 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | would_reject_rate | 0.10000000000000009 |
| vectorized_4env | 005_pure_blended_soft_geometry_light_alpha_narrow_scope_adam_minus_005_pure_blended_baseline_adam | backbone_active_preconditioner_p99 | -549517.0 |