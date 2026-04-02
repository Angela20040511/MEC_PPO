"""Runtime safety guards for configuration and checkpoint handling."""

from __future__ import annotations

from config import MECConfig


LEGACY_POLICY_RATIO_MODES = {
    "current_joint_sum_ratio",
}


def resolve_policy_ratio_mode(raw_mode: str) -> str:
    """Resolve alias policy modes to their canonical implementation mode."""
    alias_to_base_mode = {
        "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate": (
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
        ),
        "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1": (
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
        ),
        "hierarchical_actor_route_step_alignment_gate": (
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
        ),
    }
    return alias_to_base_mode.get(raw_mode, raw_mode)


def validate_runtime_config(
    config: MECConfig,
    *,
    topology_mode: str,
    allow_legacy_mode: bool = False,
) -> None:
    """Fail fast when a runtime config is likely to run an unintended legacy policy mode."""
    raw_mode = config.ppo.policy_ratio_mode
    if allow_legacy_mode:
        return
    if topology_mode != "dense":
        return
    if raw_mode in LEGACY_POLICY_RATIO_MODES:
        resolved_mode = resolve_policy_ratio_mode(raw_mode)
        raise ValueError(
            "Blocked legacy dense policy_ratio_mode. "
            f"raw_mode={raw_mode}, resolved_mode={resolved_mode}. "
            "Use --allow-legacy-mode to bypass this safety check intentionally."
        )
