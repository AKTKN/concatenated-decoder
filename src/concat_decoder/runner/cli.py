from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ..config import DecoderConfig, ThresholdExperimentConfig
from .experiment import run_threshold_experiment
from ..utils.visualization import threshold_plot_paths


def _load_config(path: Path) -> tuple[ThresholdExperimentConfig, DecoderConfig]:
    """Loads experiment and decoder configurations from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_exp = dict(data.get("experiment", {}))

    if "circuit_kind" in raw_exp and "circuit_style" not in raw_exp:
        kind = raw_exp.pop("circuit_kind")
        if kind == "superdense":
            raw_exp["circuit_style"] = "superdense_color_code_{basis}"
        elif kind == "midout":
            raw_exp["circuit_style"] = "midout_color_code_{basis}"
        else:
            raise ValueError(f"Unknown legacy circuit_kind: {kind}")

    exp_cfg = ThresholdExperimentConfig.from_dict(raw_exp)
    dec_cfg = DecoderConfig.from_dict(data.get("decoder", {}))
    return exp_cfg, dec_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantum memory experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/memory_simulation.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Override total shots in config.",
    )
    parser.add_argument(
        "--shots-per-batch",
        type=int,
        default=None,
        help="Override chunk size per sampling batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker process count. Use 1 for sequential.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs.",
    )
    args = parser.parse_args()

    exp_cfg, dec_cfg = _load_config(args.config)

    if (
        args.shots is not None
        or args.shots_per_batch is not None
        or args.workers is not None
        or args.quiet
    ):
        if bool(getattr(getattr(exp_cfg, "exhaustive_check", None), "enabled", False)):
            if args.shots is not None or args.shots_per_batch is not None:
                print("[concat-decoder] NOTE: --shots/--shots-per-batch are ignored in exhaustive_check mode")
            if args.workers is not None:
                print("[concat-decoder] NOTE: --workers is ignored in exhaustive_check mode")
            exp_cfg = replace(
                exp_cfg,
                **({"verbose": False} if args.quiet else {}),
            )
        else:
            exp_cfg = replace(
                exp_cfg,
                **({"shots": int(args.shots)} if args.shots is not None else {}),
                **({"shots_per_batch": int(args.shots_per_batch)} if args.shots_per_batch is not None else {}),
                **({"workers": int(args.workers)} if args.workers is not None else {}),
                **({"verbose": False} if args.quiet else {}),
            )

    if bool(getattr(getattr(exp_cfg, "exhaustive_check", None), "enabled", False)):
        from .exhaustive_check import run_exhaustive_check
        summary = run_exhaustive_check(exp_cfg, dec_cfg)
        out_dir = Path(exp_cfg.output_dir)
        print("\n[concat-decoder] Finished exhaustive_check")
        print(
            "[concat-decoder] "
            f"d={summary.distance} d2={summary.d2} p={summary.p:.12g} w={summary.weight} "
            f"patterns={summary.num_patterns} fail={summary.num_fail} abort={summary.num_abort}"
        )
        print(f"[concat-decoder] Output Dir: {out_dir}")
        return

    rows = run_threshold_experiment(exp_cfg, dec_cfg)

    out_dir = Path(exp_cfg.output_dir)
    plot_paths = threshold_plot_paths(out_dir, exp_cfg.plot_ler_unit, exp_cfg.plot_x_scale)

    print(f"\n[concat-decoder] Finished memory experiment: {len(rows)} points evaluated.")
    print(f"[concat-decoder] Results CSV: {out_dir / 'threshold_results.csv'}")

    if dec_cfg.hybrid_two_stage.detailed_stats:
        print(f"[concat-decoder] Detailed Stats: {out_dir / 'detailed_stats.parquet'}")

    if bool(getattr(exp_cfg.detailed_stats, "enabled", False)):
        subdir = str(getattr(exp_cfg.detailed_stats, "output_subdir", "detailed_stats"))
        print(f"[concat-decoder] Shot-level Detailed Stats Dir: {out_dir / subdir}")

    print("[concat-decoder] Generated Plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
