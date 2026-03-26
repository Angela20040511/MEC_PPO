# Critic Fit Sanity Check

- baseline_mode: `hierarchical_actor_joint_reward_aligned_credit`
- checkpoint_path: `checkpoints\dense_policy_joint_td_aligned_credit_experiment_20260321_182900\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_182909_053940\best_model.pt`
- fit_steps: 100
- buffer_steps: 100

## Group A
- initial loss: 1.728728
- final loss: 0.000030
- final pred-target pearson: 1.0000
- final pred-return pearson: 1.0000

## Group B
- initial loss: 1.245985
- final loss: 0.000147
- final pred-target pearson: 0.9999
- final pred-return pearson: -0.9999

## Comparison
- easier_group: A_value_target
- group_a_loss_drop: 1.728698
- group_b_loss_drop: 1.245838
