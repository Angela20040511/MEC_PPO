# Route Credit Path Audit

- shared checkpoint: `checkpoints\dense_policy_route_candidate_score_credit_experiment_20260320_111331\policy_ratio_hierarchical_actor_true_conditional_route_policy_20260320_111334_884525\best_model.pt`
- rollout seed: `2025`
- fixed minibatch size: `32`
- fixed minibatch source: `single rollout collected once from the shared baseline checkpoint, then replayed through both audit modes with identical batch indices`

## Conclusion

- case: `3`
- diagnosis: 裸 backward 和完整控制逻辑都正常，单 minibatch 路径已走通，更像是完整训练中的动态因素。

## Key Results

| mode | bare route head grad | bare route backbone grad | full route update_count | optimizer.step | route params changed |
|---|---:|---:|---:|---:|---:|
| hierarchical_actor_true_conditional_route_policy | 0.127929 | 0.124472 | 1 | 1 | 1 |
| hierarchical_actor_true_conditional_route_candidate_score_credit | 0.125369 | 0.140193 | 1 | 1 | 1 |
