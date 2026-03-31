from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
import numpy as np

from .config import StatisticalErrorConfig


@dataclass(frozen=True)
class FailureStats:
    pfail: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _wilson_interval(num_fails: np.ndarray, shots: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if shots <= 0:
        raise ValueError("shots must be positive")

    k = np.asarray(num_fails, dtype=float)
    n = float(shots)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denom

    low = np.clip(center - half, 0.0, 1.0)
    high = np.clip(center + half, 0.0, 1.0)
    return low, high


_CONFINT_METHODS = {
    "wilson": _wilson_interval,
}


def compute_failure_stats(
    num_fails: np.ndarray | int,
    shots: int,
    cfg: StatisticalErrorConfig,
) -> FailureStats:
    pfail = np.asarray(num_fails, dtype=float) / float(shots)
    if not cfg.enabled:
        nan = np.full_like(pfail, np.nan, dtype=float)
        return FailureStats(pfail=pfail, ci_low=nan, ci_high=nan)

    method = str(cfg.method).strip().lower()
    if method not in _CONFINT_METHODS:
        raise ValueError(f"Unsupported confint method: {cfg.method}")

    low, high = _CONFINT_METHODS[method](num_fails, shots, float(cfg.alpha))
    return FailureStats(pfail=pfail, ci_low=low, ci_high=high)


def scale_interval(low: np.ndarray, high: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    return low * scale, high * scale
