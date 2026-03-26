# Counterfactual Value Semantics Audit

- baseline_mode: `hierarchical_actor_joint_reward_aligned_credit`
- checkpoint_path: `checkpoints\dense_policy_joint_td_aligned_credit_experiment_20260321_182900\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_182909_053940\best_model.pt`
- sample_count: 24
- block_count: 24
- row_count: 1728

## Value vs Objective
- selected value vs return: pearson=-0.3094, spearman=-0.3787
- selected value vs advantage: pearson=-0.1765, spearman=-0.3661
- best value vs return: pearson=-0.3005, spearman=-0.3589

## Value vs Reward
- selected value vs selected reward: pearson=-0.1585, spearman=-0.0962
- best value vs best reward: pearson=-0.1612, spearman=-0.1054

## Ranking Flips
- reward best vs value best match ratio: 0.5330

## Distribution Shift
- actual next-state value mean/std: -21.5118 / 8.9436
- counterfactual next-state value mean/std: -20.9722 / 9.2202
- raw state distance mean/std: 0.0941 / 0.0897
- critic input distance mean/std: 2.5831 / 2.3932

## Top Flip Types
- reward:local->value:bs2: 315
- reward:local->value:bs1: 174
- reward:bs1->value:bs2: 153
- reward:bs2->value:local: 81
- reward:bs1->value:local: 63
- reward:bs2->value:bs1: 21
