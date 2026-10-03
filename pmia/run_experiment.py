"""Command-line interface for the corrected experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from .experiments import (
        SUPPORTED_METHODS,
        SUPPORTED_MODALITIES,
        SUPPORTED_MODELS,
        preset_settings,
        print_result,
        run_experiment,
        save_result,
    )
else:
    from experiments import (
        SUPPORTED_METHODS,
        SUPPORTED_MODALITIES,
        SUPPORTED_MODELS,
        preset_settings,
        print_result,
        run_experiment,
        save_result,
    )


def _methods(value: str, modality: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "all":
        methods = ["baseline", "noise", "mask", "feature_selection"]
        if modality == "purchase":
            methods.append("enumerate")
        return tuple(methods)
    methods = tuple(part.strip() for part in normalized.split(",") if part.strip())
    if not methods:
        raise argparse.ArgumentTypeError("--methods must not be empty")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods {unknown}; choose from {SUPPORTED_METHODS} or 'all'"
        )
    return methods


def build_parser(
    *,
    preset_modality: str | None = None,
    preset_model: str | None = None,
) -> argparse.ArgumentParser:
    description = (
        "Run a validated partial-membership inference experiment. Any missing "
        "dependency, invalid metric, convergence warning, or memory-limit breach "
        "causes a nonzero failure."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--modality",
        choices=SUPPORTED_MODALITIES,
        default=preset_modality,
        required=preset_modality is None,
    )
    parser.add_argument(
        "--model",
        dest="model_family",
        choices=SUPPORTED_MODELS,
        default=preset_model,
        required=preset_model is None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Explicitly run a reduced 5,000-record configuration. Results are "
            "labeled SMOKE/REDUCED."
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated methods: baseline,noise,mask,feature_selection,"
            "enumerate; use 'all' for every method supported by the modality. "
            "Enumeration is opt-in because it grows exponentially."
        ),
    )
    parser.add_argument("--training-size", type=int, default=None)
    parser.add_argument("--testing-size", type=int, default=None)
    parser.add_argument("--evaluation-size", type=int, default=None)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--target-fraction", type=float, default=None)
    parser.add_argument("--training-proportion", type=float, default=None)
    parser.add_argument("--shadow-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--clusters",
        type=int,
        choices=(2, 10, 20, 50, 100),
        default=None,
        help="Purchase label-cluster count; ignored modalities reject non-default use.",
    )
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--feature-fraction", type=float, default=None)
    parser.add_argument("--enumeration-fraction", type=float, default=None)
    parser.add_argument("--mask-value", type=float, default=None)
    parser.add_argument("--max-output-mib", type=int, default=None)
    parser.add_argument("--target-max-iter", type=int, default=None)
    parser.add_argument("--attack-max-iter", type=int, default=None)
    parser.add_argument(
        "--no-hash-data",
        action="store_true",
        help="Explicitly skip potentially expensive data hashing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Atomically write the complete JSON result to this path.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    preset_modality: str | None = None,
    preset_model: str | None = None,
) -> int:
    parser = build_parser(
        preset_modality=preset_modality,
        preset_model=preset_model,
    )
    args = parser.parse_args(argv)
    selected_methods = (
        None if args.methods is None else _methods(args.methods, args.modality)
    )
    max_output_bytes = (
        None
        if args.max_output_mib is None
        else args.max_output_mib * 1024 * 1024
    )
    settings = preset_settings(
        args.modality,
        args.model_family,
        smoke=args.smoke,
        training_size=args.training_size,
        testing_size=args.testing_size,
        evaluation_size=args.evaluation_size,
        dataset_limit=args.dataset_limit,
        target_fraction=args.target_fraction,
        training_proportion=args.training_proportion,
        num_shadow_models=args.shadow_models,
        seed=args.seed,
        clusters=args.clusters,
        noise_level=args.noise_level,
        feature_fraction=args.feature_fraction,
        enumeration_fraction=args.enumeration_fraction,
        mask_value=args.mask_value,
        max_output_bytes=max_output_bytes,
        target_max_iter=args.target_max_iter,
        attack_max_iter=args.attack_max_iter,
        methods=selected_methods,
        hash_data=False if args.no_hash_data else None,
    )
    result = run_experiment(settings)
    if args.output is not None:
        save_result(result, args.output)
    print_result(result)
    return 0


def preset_entrypoint(
    modality: str,
    model_family: str,
    argv: Sequence[str] | None = None,
) -> int:
    return main(
        argv,
        preset_modality=modality,
        preset_model=model_family,
    )


if __name__ == "__main__":
    raise SystemExit(main())
