from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .config import ThresholdPoint, PlotLerUnit, PlotXScale

def plot_threshold(
    rows: list[ThresholdPoint],
    basis: str,
    output_png: Path,
    unit: str,
    x_scale: str,
) -> None:
    by_pair: dict[tuple[int, int | None], list[ThresholdPoint]] = {}
    for r in rows:
        by_pair.setdefault((int(r.distance), getattr(r, "d2", None)), []).append(r)

    fig, ax = plt.subplots(figsize=(8, 6))
    def _pair_sort_key(k: tuple[int, int | None]) -> tuple[int, int]:
        d, d2 = k
        return (int(d), -1 if d2 is None else int(d2))

    for (d, d2) in sorted(by_pair.keys(), key=_pair_sort_key):
        pts = sorted(by_pair[(d, d2)], key=lambda x: x.p)
        x = np.array([t.p for t in pts])
        y = np.array([t.ler_x if basis == "X" else t.ler_z for t in pts])
        y_low = np.array([t.ler_x_ci_low if basis == "X" else t.ler_z_ci_low for t in pts])
        y_high = np.array([t.ler_x_ci_high if basis == "X" else t.ler_z_ci_high for t in pts])
        if unit == "per_shot":
            rounds = np.array([max(1, int(t.effective_rounds)) for t in pts], dtype=float)
            y = y * rounds
            y_low = y_low * rounds
            y_high = y_high * rounds
        y = np.maximum(y, 1e-12)
        y_low = np.maximum(y_low, 1e-12)
        y_high = np.maximum(y_high, y_low)

        line = ax.plot(x, y, marker="o", label=f"(d,d2)=({d},{d2})")[0]
        color = line.get_color()
        mask = np.isfinite(y_low) & np.isfinite(y_high)
        if np.any(mask):
            ax.fill_between(x[mask], y_low[mask], y_high[mask], color=color, alpha=0.2, linewidth=0.0)
            yerr = np.vstack([y[mask] - y_low[mask], y_high[mask] - y[mask]])
            ax.errorbar(x[mask], y[mask], yerr=yerr, fmt="none", ecolor=color, alpha=0.6, capsize=3)

    if x_scale not in {"log", "linear"}:
        raise ValueError("x_scale must be 'log' or 'linear'")
    ax.set_xscale(x_scale)
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    unit_label = "per round" if unit == "per_round" else "per shot"
    ax.set_ylabel(f"Logical error rate ({unit_label})")
    ax.set_title(f"Threshold sweep ({basis} basis)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

def plot_thresholds(
    rows: list[ThresholdPoint],
    basis: str,
    output_dir: Path,
    plot_ler_unit: PlotLerUnit,
    plot_x_scale: PlotXScale,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for unit, x_scale, out in _iter_threshold_plot_targets(output_dir, plot_ler_unit, plot_x_scale):
        plot_threshold(rows, basis, out, unit, x_scale)
        created.append(out)

    return created


def threshold_plot_paths(
    output_dir: Path,
    plot_ler_unit: PlotLerUnit,
    plot_x_scale: PlotXScale,
) -> list[Path]:
    return [p for _, _, p in _iter_threshold_plot_targets(output_dir, plot_ler_unit, plot_x_scale)]


def _iter_threshold_plot_targets(
    output_dir: Path,
    plot_ler_unit: PlotLerUnit,
    plot_x_scale: PlotXScale,
) -> list[tuple[str, str, Path]]:
    units = _resolve_plot_ler_units(plot_ler_unit)
    x_scales = _resolve_plot_x_scales(plot_x_scale)
    targets: list[tuple[str, str, Path]] = []
    for unit in units:
        base = "threshold_per_round" if unit == "per_round" else "threshold_per_shot"
        for x_scale in x_scales:
            suffix = "" if x_scale == "log" else f"_{x_scale}"
            targets.append((unit, x_scale, output_dir / f"{base}{suffix}.png"))
    return targets


def _resolve_plot_ler_units(plot_ler_unit: PlotLerUnit) -> list[str]:
    mode = str(plot_ler_unit)
    if mode == "per_round":
        return ["per_round"]
    if mode == "per_shot":
        return ["per_shot"]
    if mode == "both":
        return ["per_round", "per_shot"]
    raise ValueError("plot_ler_unit must be one of: per_round, per_shot, both")


def _resolve_plot_x_scales(plot_x_scale: PlotXScale) -> list[str]:
    mode = str(plot_x_scale)
    if mode == "log":
        return ["log"]
    if mode == "linear":
        return ["linear"]
    if mode == "both":
        return ["log", "linear"]
    raise ValueError("plot_x_scale must be one of: log, linear, both")