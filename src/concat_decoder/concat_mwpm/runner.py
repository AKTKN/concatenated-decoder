from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .path_setup import ensure_color_code_stim_import


@dataclass(frozen=True)
class Task:
    decoder_variant: str
    distance: int
    rounds: int
    physical_error_rate: float


def _build_noise_model(noise_cfg: dict[str, Any], p: float):
    mod = importlib.import_module("color_code_stim")
    NoiseModel = mod.NoiseModel

    mode = str(noise_cfg.get("mode", "uniform_circuit_noise"))
    fixed_kwargs = dict(noise_cfg.get("fixed_kwargs", {}))

    if mode == "uniform_circuit_noise":
        return NoiseModel.uniform_circuit_noise(p)

    if mode == "bitflip":
        key = str(noise_cfg.get("rate_key", "bitflip"))
        kwargs = {**fixed_kwargs, key: p}
        return NoiseModel(**kwargs)

    if mode == "noise_model_kwargs":
        key = str(noise_cfg.get("rate_key", "bitflip"))
        kwargs = {**fixed_kwargs, key: p}
        return NoiseModel(**kwargs)

    raise ValueError(
        "Unknown noise mode. Use one of: "
        "uniform_circuit_noise, bitflip, noise_model_kwargs"
    )


def _run_task(
    task: Task,
    circuit_type: str,
    shots: int,
    noise_cfg: dict[str, Any],
    color_code_kwargs: dict[str, Any],
    bplsd_prior_boost_factor: float,
    seed: int | None,
) -> dict[str, Any]:
    ensure_color_code_stim_import()
    mod = importlib.import_module("color_code_stim")
    ColorCode = mod.ColorCode

    noise_model = _build_noise_model(noise_cfg, task.physical_error_rate)

    color_code = ColorCode(
        d=task.distance,
        rounds=task.rounds,
        circuit_type=circuit_type,
        noise_model=noise_model,
        **color_code_kwargs,
    )

    simulate_kwargs: dict[str, Any] = {
        "shots": shots,
        "concat_unionfind": False,
        "full_output": True,
        "verbose": False,
    }

    if seed is not None:
        simulate_kwargs["seed"] = int(seed)

    if task.decoder_variant == "mwpm":
        pass
    elif task.decoder_variant == "uf":
        simulate_kwargs["concat_unionfind"] = True
    elif task.decoder_variant == "mwpm_bplsd":
        simulate_kwargs["bplsd_refine_from_color_candidates"] = True
        simulate_kwargs["bplsd_prior_boost_factor"] = float(bplsd_prior_boost_factor)
    else:
        raise ValueError(f"Unsupported decoder variant: {task.decoder_variant}")

    _, info = color_code.simulate(**simulate_kwargs)
    pfail, delta_pfail = info["stats"]

    ler_per_round = float(pfail) / float(task.rounds)
    delta_ler_per_round = float(delta_pfail) / float(task.rounds)

    return {
        "decoder_variant": task.decoder_variant,
        "distance": int(task.distance),
        "rounds": int(task.rounds),
        "physical_error_rate": float(task.physical_error_rate),
        "pfail": float(pfail),
        "delta_pfail": float(delta_pfail),
        "ler_per_round": ler_per_round,
        "delta_ler_per_round": delta_ler_per_round,
        "shots": int(shots),
    }


def _resolve_rounds(rounds_spec: int | str, distance: int) -> int:
    if isinstance(rounds_spec, int):
        return rounds_spec
    return distance


def _tasks_from_config(cfg: ExperimentConfig) -> list[Task]:
    tasks: list[Task] = []
    for variant in cfg.decoder_variants:
        for d in cfg.distances:
            rounds = _resolve_rounds(cfg.rounds, d)
            for p in cfg.physical_error_rates:
                tasks.append(
                    Task(
                        decoder_variant=str(variant),
                        distance=int(d),
                        rounds=int(rounds),
                        physical_error_rate=float(p),
                    )
                )
    return tasks


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "decoder_variant",
        "distance",
        "rounds",
        "physical_error_rate",
        "pfail",
        "delta_pfail",
        "ler_per_round",
        "delta_ler_per_round",
        "shots",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    str(r["decoder_variant"]),
                    str(r["distance"]),
                    str(r["rounds"]),
                    f"{float(r['physical_error_rate']):.12g}",
                    f"{float(r['pfail']):.12g}",
                    f"{float(r['delta_pfail']):.12g}",
                    f"{float(r['ler_per_round']):.12g}",
                    f"{float(r['delta_ler_per_round']):.12g}",
                    str(r["shots"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _plot_ler_per_round(rows: list[dict[str, Any]], out_png: Path) -> None:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for r in rows:
        variant = str(r["decoder_variant"])
        distance = int(r["distance"])
        grouped.setdefault(variant, {}).setdefault(distance, []).append(r)

    ncols = max(len(grouped), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for ax, (variant, per_d) in zip(axes, sorted(grouped.items(), key=lambda x: x[0])):
        for i, d in enumerate(sorted(per_d.keys())):
            points = sorted(per_d[d], key=lambda x: float(x["physical_error_rate"]))
            x = np.array([float(p["physical_error_rate"]) for p in points], dtype=float)
            y = np.array([float(p["ler_per_round"]) for p in points], dtype=float)
            yerr = np.array([float(p["delta_ler_per_round"]) for p in points], dtype=float)

            y = y.copy()
            min_positive = 1.0 / (2.0 * float(points[0]["shots"]) * float(points[0]["rounds"]))
            y[y <= 0] = min_positive

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=f"{markers[i % len(markers)]}-",
                color=colors[i % len(colors)],
                capsize=3,
                label=f"d={d}",
            )

        ax.set_title(f"Decoder: {variant}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate p")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    axes[0].set_ylabel("LER per round")
    fig.suptitle("Memory experiment (color-code-stim): LER per round vs p")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig, quiet: bool = False) -> dict[str, str]:
    ensure_color_code_stim_import()

    out_dir = Path(cfg.output_dir).resolve() / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_out_path = out_dir / "resolved_config.json"
    if cfg_out_path.exists() and not cfg.overwrite_outputs:
        raise FileExistsError(
            f"Output already exists: {cfg_out_path}. Set overwrite_outputs=true to replace."
        )

    cfg_out_path.write_text(json.dumps(cfg.__dict__, indent=2, sort_keys=True) + "\n")

    tasks = _tasks_from_config(cfg)
    rows: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        future_map = {}
        for i, t in enumerate(tasks):
            seed = None if cfg.base_seed is None else int(cfg.base_seed) + i
            f = ex.submit(
                _run_task,
                t,
                cfg.circuit_type,
                cfg.shots,
                cfg.noise_model,
                cfg.color_code_kwargs,
                cfg.bplsd_prior_boost_factor,
                seed,
            )
            future_map[f] = t

        total = len(future_map)
        done = 0
        for f in as_completed(future_map):
            rows.append(f.result())
            done += 1
            if not quiet:
                t = future_map[f]
                print(
                    f"[{done}/{total}] {t.decoder_variant} d={t.distance} "
                    f"p={t.physical_error_rate:.3e} done"
                )

    rows.sort(
        key=lambda r: (
            str(r["decoder_variant"]),
            int(r["distance"]),
            float(r["physical_error_rate"]),
        )
    )

    csv_path = out_dir / "memory_results.csv"
    _write_results_csv(csv_path, rows)

    plot_path = out_dir / "ler_per_round_vs_physical_error_rate.png"
    _plot_ler_per_round(rows, plot_path)

    return {
        "output_dir": str(out_dir),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "config": str(cfg_out_path),
    }
