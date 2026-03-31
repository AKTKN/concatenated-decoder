from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run memory experiment using color-code-stim decoders "
            "and output LER-per-round plot."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config JSON",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    outputs = run_experiment(cfg, quiet=bool(args.quiet))

    print("[concat_mwpm] completed")
    print(f"  output dir : {outputs['output_dir']}")
    print(f"  csv        : {outputs['csv']}")
    print(f"  plot       : {outputs['plot']}")
    print(f"  config     : {outputs['config']}")


if __name__ == "__main__":
    main()
