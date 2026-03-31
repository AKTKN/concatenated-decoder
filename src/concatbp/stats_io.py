from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
from .config import ThresholdPoint, ThresholdExperimentConfig, DecoderConfig


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "none" or s.lower() == "nan":
        return None
    return int(s)

def write_detailed_stats_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    try:
        import pyarrow # noqa: F401
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write detailed stats. pip install pyarrow") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def load_existing_points(path: Path) -> set[tuple[int, int | None, float]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int | None, float]] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("ler_unit", "") == "per_round":
                keys.add((int(r["distance"]), _parse_optional_int(r.get("d2")), float(r["p"])))
    return keys


def read_threshold_csv(path: Path) -> list[ThresholdPoint]:
    if not path.exists():
        return []
    rows: list[ThresholdPoint] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ThresholdPoint(
                    distance=int(r["distance"]),
                    d2=_parse_optional_int(r.get("d2")),
                    p=float(r["p"]),
                    style=str(r.get("style", "")),
                    noise_model=str(r.get("noise_model", "")),
                    ler_x=float(r["ler_x"]),
                    ler_z=float(r["ler_z"]),
                    ler_x_ci_low=float(r.get("ler_x_ci_low", "nan")),
                    ler_x_ci_high=float(r.get("ler_x_ci_high", "nan")),
                    ler_z_ci_low=float(r.get("ler_z_ci_low", "nan")),
                    ler_z_ci_high=float(r.get("ler_z_ci_high", "nan")),
                    shots=int(r["shots"]),
                    rounds_spec=str(r.get("rounds_spec", "")),
                    requested_rounds=int(r.get("requested_rounds", r.get("effective_rounds", 0))),
                    effective_rounds=int(r.get("effective_rounds", 0)),
                    abort_count=int(r.get("abort_count", 0) or 0),
                    abort_rate=float(r.get("abort_rate", 0.0) or 0.0),
                    postselected_ler_x=float(r.get("postselected_ler_x", "nan")),
                    postselected_ler_z=float(r.get("postselected_ler_z", "nan")),
                )
            )
    return rows

def merge_existing_rows(path: Path, new_rows: list[ThresholdPoint]) -> list[ThresholdPoint]:
    merged: dict[tuple[int, int | None, float], ThresholdPoint] = {}
    for r in read_threshold_csv(path):
        merged[(r.distance, r.d2, r.p)] = r
    for r in new_rows:
        merged[(r.distance, r.d2, r.p)] = r
    return [
        merged[k]
        for k in sorted(
            merged.keys(),
            key=lambda x: (int(x[0]), -1 if x[1] is None else int(x[1]), float(x[2])),
        )
    ]

def write_threshold_csv(rows: list[ThresholdPoint], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "distance", "d2", "p", "style", "noise_model", "ler_x", "ler_z",
                "ler_x_ci_low", "ler_x_ci_high", "ler_z_ci_low", "ler_z_ci_high",
                "ler_unit", "shots", "rounds_spec", "requested_rounds", "effective_rounds",
                "abort_count", "abort_rate", "postselected_ler_x", "postselected_ler_z",
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "distance": r.distance, "d2": "" if r.d2 is None else int(r.d2), "p": r.p, "style": r.style,
                "noise_model": r.noise_model, "ler_x": r.ler_x, "ler_z": r.ler_z,
                "ler_x_ci_low": r.ler_x_ci_low, "ler_x_ci_high": r.ler_x_ci_high,
                "ler_z_ci_low": r.ler_z_ci_low, "ler_z_ci_high": r.ler_z_ci_high,
                "ler_unit": "per_round", "shots": r.shots, "rounds_spec": r.rounds_spec,
                "requested_rounds": r.requested_rounds, "effective_rounds": r.effective_rounds,
                "abort_count": int(getattr(r, "abort_count", 0) or 0),
                "abort_rate": float(getattr(r, "abort_rate", 0.0) or 0.0),
                "postselected_ler_x": float(getattr(r, "postselected_ler_x", float("nan"))),
                "postselected_ler_z": float(getattr(r, "postselected_ler_z", float("nan"))),
            })

def write_meta(cfg: ThresholdExperimentConfig, decoder_cfg: DecoderConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for k, v in cfg.__dict__.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        for k, v in decoder_cfg.__dict__.items():
            f.write(f"{k}: {v}\n")

