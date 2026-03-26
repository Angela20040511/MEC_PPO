# Critic Target Pipeline Trace

1. Rollout collection
   - `select_action(state)` returns raw `value = evaluate_value(state)`.
   - Buffer stores `reward`, `done`, `value`, `state`, `next_state`.

2. Trajectory finish
   - `delta_t = reward_t + gamma * value_{t+1} * (1-done_t) - value_t`
   - `gae_t = delta_t + gamma * gae_lambda * (1-done_t) * gae_{t+1}`
   - `return_t = gae_t + value_t`

3. Training target under `popart_return_norm`
   - target_mean = -25.268698
   - target_std = 7.841250
   - `value_targets = (returns - target_mean) / (target_std + 1e-8)`

4. Critic prediction used in loss
   - `normalized_values = network.normalized_value_from_critic_input(critic_input)`
   - loss fits `normalized_values` against `value_targets`

5. Inference-time value
   - `evaluate_value(state)` returns raw de-normalized value
   - equivalent normalized prediction is `(evaluate_value(state) - target_mean) / target_std`
