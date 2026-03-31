from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    distances: list[int]
    physical_error_rates: list[float]
    shots: int
    rounds: int | str = "d"
    circuit_type: str = "tri"
    workers: int | None = None
    base_seed: int | None = 42
    decoder_variants: list[str] = field(default_factory=lambda: ["mwpm", "uf"])
    noise_model: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "uniform_circuit_noise",
            "rate_key": "bitflip",
            "fixed_kwargs": {},
        }
    )
    color_code_kwargs: dict[str, Any] = field(default_factory=dict)
    bplsd_prior_boost_factor: float = 50.0
    overwrite_outputs: bool = True


def _parse_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_distances(raw: Any) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("distances must be a non-empty list")
    parsed: list[int] = []
    for v in raw:
        parsed.append(_parse_positive_int(v, "distance"))
    return parsed


def _resolve_rates(raw: Any) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("physical_error_rates must be a non-empty list")
    vals: list[float] = []
    for v in raw:
        p = float(v)
        if p <= 0:
            raise ValueError("physical_error_rates must contain positive values")
        vals.append(p)
    return vals


def _resolve_rounds(raw: Any) -> int | str:
    if isinstance(raw, int):
        return _parse_positive_int(raw, "rounds")
    if isinstance(raw, str):
        token = raw.strip().lower().replace(" ", "")
        if token == "d":
            return "d"
        if token.isdigit() and int(token) > 0:
            return int(token)
    raise ValueError("rounds must be a positive integer or 'd'")


def load_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text())

    experiment_name = str(raw.get("experiment_name", "concat_mwpm_memory"))
    output_dir = str(raw.get("output_dir", "concatbp_outputs/concat_mwpm"))
    shots = _parse_positive_int(raw.get("shots", 1000), "shots")

    workers_raw = raw.get("workers", None)
    workers = None if workers_raw is None else _parse_positive_int(workers_raw, "workers")

    base_seed_raw = raw.get("base_seed", 42)
    base_seed = None if base_seed_raw is None else int(base_seed_raw)

    decoder_variants = list(raw.get("decoder_variants", ["mwpm", "uf"]))
    if not decoder_variants:
        raise ValueError("decoder_variants must be non-empty")

    noise_model = dict(raw.get("noise_model", {}))
    color_code_kwargs = dict(raw.get("color_code_kwargs", {}))

    bplsd_prior_boost_factor = float(raw.get("bplsd_prior_boost_factor", 50.0))
    if bplsd_prior_boost_factor <= 0:
        raise ValueError("bplsd_prior_boost_factor must be positive")

    overwrite_outputs = bool(raw.get("overwrite_outputs", True))

    return ExperimentConfig(
        experiment_name=experiment_name,
        output_dir=output_dir,
        distances=_resolve_distances(raw.get("distances")),
        physical_error_rates=_resolve_rates(raw.get("physical_error_rates")),
        shots=shots,
        rounds=_resolve_rounds(raw.get("rounds", "d")),
        circuit_type=str(raw.get("circuit_type", "tri")),
        workers=workers,
        base_seed=base_seed,
        decoder_variants=decoder_variants,
        noise_model=noise_model,
        color_code_kwargs=color_code_kwargs,
        bplsd_prior_boost_factor=bplsd_prior_boost_factor,
        overwrite_outputs=overwrite_outputs,
    )
