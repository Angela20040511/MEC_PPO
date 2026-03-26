# Joint Credit Reward Semantics Audit

- baseline_mode: `hierarchical_actor_true_conditional_route_policy`
- checkpoint_path: `checkpoints\dense_policy_true_conditional_route_policy_experiment_20260320_084754\policy_ratio_hierarchical_actor_true_conditional_route_policy_20260320_084850_383994\best_model.pt`
- sample_count: 100
- block_count: 24
- row_count: 2400

## Joint Score vs Objective
- selected_joint_score vs advantage: pearson=-0.0077, spearman=-0.0024
- selected_joint_score vs return: pearson=-0.0151, spearman=0.0083
- best_joint_score vs advantage: pearson=-0.0228, spearman=-0.0170
- best_joint_score vs return: pearson=-0.0062, spearman=0.0104

## Joint Action Agreement
- joint_action_decision_agreement_ratio: 0.4892
- actual_local_rate_when_best_action_local: 0.4900
- actual_bs1_rate_when_best_action_bs1: 0.0000
- actual_bs2_rate_when_best_action_bs2: 0.0000

## Branch Credit Semantics
- A_theta vs theta_true_gap: pearson=-0.1619, spearman=-0.2042, sign_agreement=0.5121
- A_route vs route_true_gap: pearson=-0.0913, spearman=-0.1286, sign_agreement=0.4334
- joint_action_induced_by_A_match_ratio: 0.5121

## Path Cost vs Reward Conflict
- path_cost_best_action_has_negative_advantage_ratio: 0.8765
- path_cost_best_action_has_negative_return_ratio: 1.0000
- path_cost_suboptimal_action_has_positive_advantage_ratio: 0.1166
- path_cost_suboptimal_action_has_positive_return_ratio: 0.0000
- actual_offload_rate_when_path_cost_prefers_local: 0.5100
- actual_bs1_rate_when_path_cost_prefers_bs2: 0.0000
- actual_bs2_rate_when_path_cost_prefers_bs1: 0.3333
