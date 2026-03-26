# Reward Aligned Joint Reference Audit

- baseline_mode: `hierarchical_actor_true_conditional_route_policy`
- checkpoint_path: `checkpoints\dense_policy_true_conditional_route_policy_experiment_20260320_084754\policy_ratio_hierarchical_actor_true_conditional_route_policy_20260320_084850_383994\best_model.pt`
- sample_count: 24
- block_count: 24
- row_count: 576

## Joint Score vs Objective
- path selected vs advantage: pearson=-0.0154, spearman=-0.0462
- reward-aligned selected vs advantage: pearson=0.1082, spearman=0.1825
- path selected vs return: pearson=0.0162, spearman=-0.0036
- reward-aligned selected vs return: pearson=0.2624, spearman=0.2875

## Joint Action Agreement
- under path: 0.5139
- under reward-aligned: 0.4253

## Reference Conflict
- path_best_vs_reward_aligned_best_match_ratio: 0.6632

## Branch Credit vs Reward-Aligned Gaps
- A_theta vs theta_true_gap_path: pearson=-0.2149, spearman=-0.2467, sign=0.5434
- A_theta vs theta_true_gap_reward_aligned: pearson=0.0274, spearman=0.0381, sign=0.5469
- A_route vs route_true_gap_path: pearson=-0.1719, spearman=-0.2848, sign=0.3692
- A_route vs route_true_gap_reward_aligned: pearson=0.0961, spearman=0.1436, sign=0.5591
