import argparse
from pathlib import Path
from typing import Any

from dense_policy_true_conditional_route_experiment import FIXED_SEED

from critic_backbone_preconditioner_clip_telemetry_validation import run_experiment


VALIDATION_STEM = "critic_backbone_layerwise_cap_refinement_validation"
TRACKED_HOTSPOT_LAYERS = (
    "critic_backbone.0.weight",
    "critic_backbone.6.weight",
    "critic_backbone.7.weight",
    "critic_backbone.4.bias",
    "critic_backbone.7.bias",
)
HOTSPOT_CLIP_LAYERS = (
    "critic_backbone.0.weight",
    "critic_backbone.6.weight",
    "critic_backbone.7.weight",
    "critic_backbone.4.bias",
)
CURRENT_LAYERWISE_CAPS = {
    "critic_backbone.0.weight": 2e3,
    "critic_backbone.6.weight": 4e3,
    "critic_backbone.7.weight": 4e3,
    "critic_backbone.4.bias": 4e3,
}
MID_0W_LAYERWISE_CAPS = {
    "critic_backbone.0.weight": 1.5e3,
    "critic_backbone.6.weight": 4e3,
    "critic_backbone.7.weight": 4e3,
    "critic_backbone.4.bias": 4e3,
}
GROUP_SPECS: dict[str, dict[str, Any]] = {
    "baseline_critic_adam": {
        "backbone_preconditioner_cap": None,
        "layerwise_backbone_preconditioner_caps": {},
        "clip_focus_parameter_names": None,
        "description": (
            "Backbone and head both keep the current Adam rule. Split only for "
            "telemetry and exact backbone/head targeting."
        ),
    },
    "critic_backbone_layerwise_preconditioner_clip_head_adam": {
        "backbone_preconditioner_cap": None,
        "layerwise_backbone_preconditioner_caps": CURRENT_LAYERWISE_CAPS,
        "clip_focus_parameter_names": HOTSPOT_CLIP_LAYERS,
        "description": (
            "Current layerwise hotspot clip: 0.weight=2e3, "
            "6.weight/7.weight/4.bias=4e3."
        ),
    },
    "critic_backbone_layerwise_preconditioner_clip_mid_0w_head_adam": {
        "backbone_preconditioner_cap": None,
        "layerwise_backbone_preconditioner_caps": MID_0W_LAYERWISE_CAPS,
        "clip_focus_parameter_names": HOTSPOT_CLIP_LAYERS,
        "description": (
            "Only refine the 0.weight hotspot cap from 2e3 to 1.5e3 while "
            "keeping the other hotspot caps unchanged."
        ),
    },
}
DEFAULT_SELECTED_GROUPS = (
    "baseline_critic_adam",
    "critic_backbone_layerwise_preconditioner_clip_head_adam",
    "critic_backbone_layerwise_preconditioner_clip_mid_0w_head_adam",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal short validation for critic_backbone layerwise cap refinement, "
            "focused on an intermediate 0.weight cap of 1.5e3."
        )
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_SELECTED_GROUPS),
        help="Subset of validation groups to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        seed=args.seed,
        output_root=Path(args.output_root),
        group_names=list(args.groups),
        validation_stem=VALIDATION_STEM,
        group_specs=GROUP_SPECS,
        tracked_hotspot_layers=TRACKED_HOTSPOT_LAYERS,
    )
    print(
        "Critic backbone layerwise cap refinement validation completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
