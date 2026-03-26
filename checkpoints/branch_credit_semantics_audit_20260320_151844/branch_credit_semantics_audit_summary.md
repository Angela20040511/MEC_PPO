# Branch Credit Semantics Audit

- checkpoint: `checkpoints\dense_policy_route_candidate_score_credit_experiment_20260320_141920\policy_ratio_hierarchical_actor_true_conditional_route_policy_20260320_141926_045925\best_model.pt`
- baseline mode: `hierarchical_actor_true_conditional_route_policy`
- rollout samples: `100`

## Theta

- Pearson: `-0.1619`, Spearman: `-0.2042`, sign agreement: `0.5121`
- confusion: `{'tp': 2, 'fp': 1169, 'tn': 1227, 'fn': 2}`
- offload decision agreement: `0.4896`

## Route

- Pearson: `-0.0913`, Spearman: `-0.1286`, sign agreement: `0.4334`
- confusion: `{'tp': 230, 'fp': 381, 'tn': 300, 'fn': 312}`
- route decision agreement: `0.4767`
