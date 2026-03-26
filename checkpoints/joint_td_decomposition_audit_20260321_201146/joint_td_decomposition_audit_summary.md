# Joint TD Decomposition Audit

- baseline_mode: `hierarchical_actor_joint_reward_aligned_credit`
- checkpoint_path: `checkpoints\dense_policy_joint_td_aligned_credit_experiment_20260321_182900\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_182909_053940\best_model.pt`
- sample_count: 24
- block_count: 24
- row_count: 576
- gamma: 0.9900

## Correlations
- selected reward vs return: pearson=0.3476, spearman=0.3421
- selected value vs return: pearson=-0.3094, spearman=-0.3787
- selected td vs return: pearson=-0.3019, spearman=-0.3633
- selected reward vs advantage: pearson=0.2614, spearman=0.3413
- selected value vs advantage: pearson=-0.1765, spearman=-0.3661
- selected td vs advantage: pearson=-0.1707, spearman=-0.3364

## Ranking Flips
- reward best vs td best match ratio: 0.5920
- reward best vs value best match ratio: 0.5330
- value best vs td best match ratio: 0.9306

## Scale
- mean_abs_r_term: 1.6104
- mean_abs_gamma_v_term: 22.2043
- mean_abs_td_term: 23.6630
- gamma_v_over_r_abs_ratio: 14.1595
- value_dominates_ranking_flip_ratio: 0.4045
- td_margin_smaller_than_reward_margin_ratio: 0.1823

## Static Explanation Signals
- reward_td_flip_ratio: 0.4080
- selected_td_vs_return_weaker_than_reward: True
- selected_td_vs_advantage_weaker_than_reward: True
- reward_margin_on_flip_samples: 0.0289
- td_margin_collapse_signal: 1.0223

## Top Flip Types
- reward:local->td:bs2: 102
- reward:bs1->td:bs2: 51
- reward:local->td:bs1: 47
- reward:bs2->td:local: 17
- reward:bs1->td:local: 12
- reward:bs2->td:bs1: 6
