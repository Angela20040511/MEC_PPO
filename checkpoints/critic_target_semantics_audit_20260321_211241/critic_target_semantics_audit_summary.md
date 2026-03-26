# Critic Target Semantics Audit

- baseline_mode: `hierarchical_actor_joint_reward_aligned_credit`
- checkpoint_path: `checkpoints\dense_policy_joint_td_aligned_credit_experiment_20260321_182900\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_182909_053940\best_model.pt`
- sample_count: 24
- block_count: 24
- row_count: 2304

## Real Rollout Value Semantics
- value vs return: pearson=-0.3764, spearman=-0.4070
- value vs value_target: pearson=-0.3764, spearman=-0.4070
- value vs advantage: pearson=-0.9246, spearman=-0.6548
- value vs one-step reward: pearson=-0.1584, spearman=-0.2739
- value vs one-step TD target: pearson=0.0745, spearman=0.3809

## Counterfactual Value Semantics
- selected counterfactual value vs return: pearson=-0.3094, spearman=-0.3787
- selected counterfactual value vs advantage: pearson=-0.1765, spearman=-0.3661
- selected counterfactual value vs reward-aligned score: pearson=-0.1585, spearman=-0.0962

## Distribution
- real rollout state value mean/std: -21.6475 / 8.9989
- real rollout next-state value mean/std: -21.5118 / 8.9436
- counterfactual next-state value mean/std: -20.9722 / 9.2202
- raw state distance mean/std: 0.0941 / 0.0897
- critic input distance mean/std: 2.5831 / 2.3932

## Ranking Flips
- reward best vs value best match ratio: 0.5330
