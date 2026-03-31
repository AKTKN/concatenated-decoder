from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import stim


def get_obs_detector_ids_by_obs_index(circuit: stim.Circuit) -> list[int]:
    """Return detector ids corresponding to observable-detectors.

    color-code-stim encodes observable-detectors by adding a 6th coordinate entry:
        (x, y, t, pauli_val, color_val, obs_index)
    where obs_index >= 0 marks an observable-detector.

    Returns a list mapping obs_index -> det_id.
    """
    num_obs = int(circuit.num_observables)
    if num_obs <= 0:
        return []

    coords = circuit.get_detector_coordinates()
    mapping: dict[int, int] = {}
    for det_id, c in coords.items():
        if len(c) < 6:
            continue
        obs_idx = int(round(c[-1]))
        if obs_idx >= 0:
            mapping[obs_idx] = int(det_id)

    if not mapping:
        raise ValueError(
            "No observable-detectors found. comparative_decoding requires a circuit generated "
            "with color-code-stim comparative_decoding=True."
        )

    out: list[int] = []
    for k in range(num_obs):
        if k not in mapping:
            raise ValueError(
                f"Missing observable-detector for obs_index={k}. Found keys={sorted(mapping.keys())}"
            )
        out.append(int(mapping[k]))
    return out


def _all_logical_classes(num_obs: int) -> np.ndarray:
    if num_obs < 0:
        raise ValueError("num_obs must be >= 0")
    if num_obs == 0:
        return np.zeros((1, 0), dtype=bool)
    # Lexicographic order, matching itertools.product([False,True], repeat=num_obs)
    grid = np.indices((2,) * num_obs, dtype=np.int8).reshape(num_obs, -1).T
    return grid.astype(bool)


def _get_final_predictions(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Match color-code-stim's selection logic.

    Args:
        weights: shape (num_logical_classes, num_colors, num_samples)

    Returns:
        best_logical_classes: (num_samples,) int
        best_color_indices: (num_samples,) int
        weights_final: (num_samples,) float
        logical_gap: (num_samples,) float or None
    """
    if weights.ndim != 3 or weights.size == 0:
        raise ValueError(f"Invalid weights shape: {weights.shape}")

    num_logical_classes, num_colors, num_samples = weights.shape
    reshaped = weights.reshape(num_logical_classes * num_colors, num_samples)
    flat_min_indices = np.argmin(reshaped, axis=0)
    weights_final = np.min(reshaped, axis=0)

    best_logical_classes = flat_min_indices // num_colors
    best_color_indices = flat_min_indices % num_colors

    logical_gap: np.ndarray | None = None
    if num_logical_classes > 1:
        min_weights_per_class = np.min(weights, axis=1)  # min across colors
        sorted_min = np.sort(min_weights_per_class, axis=0)
        logical_gap = sorted_min[1] - sorted_min[0]

    return best_logical_classes, best_color_indices, weights_final, logical_gap


@dataclass(frozen=True)
class ComparativeDecodeResult:
    """Result of comparative decoding for a batch."""

    obs_prediction: np.ndarray
    candidate_costs: dict[str, np.ndarray]
    candidate_stage1_costs: dict[str, np.ndarray]
    decode_time_ms: np.ndarray
    logical_gap: np.ndarray


def decode_batch_comparative(
    *,
    decoder: Any,
    detector_outcomes: np.ndarray,
    obs_detector_ids_by_obs_index: Sequence[int],
    colors: Sequence[str] = ("r", "g", "b"),
) -> ComparativeDecodeResult:
    """Run comparative decoding across all logical classes.

    This function modifies the observable-detector bits in the detector outcomes to
    represent each assumed logical class, decodes each class, and selects the minimum
    weight solution.

    Requirements:
    - The circuit must have been generated with color-code-stim comparative_decoding=True.
    - The decoder must return candidate_costs and candidate_stage1_costs per color.

    Returns:
        ComparativeDecodeResult where candidate_costs and candidate_stage1_costs are
        *fixed to the selected logical class* for each shot.
    """
    dets = np.asarray(detector_outcomes, dtype=np.uint8)
    if dets.ndim == 1:
        dets = dets.reshape(1, -1)

    num_shots = int(dets.shape[0])
    num_obs = int(len(obs_detector_ids_by_obs_index))
    classes = _all_logical_classes(num_obs)
    num_classes = int(classes.shape[0])

    color_list = [str(c) for c in colors]
    num_colors = int(len(color_list))

    weights = np.full((num_classes, num_colors, num_shots), np.nan, dtype=np.float64)
    s1_weights = np.full((num_classes, num_colors, num_shots), np.nan, dtype=np.float64)

    decode_time_total = np.zeros(num_shots, dtype=np.float64)

    for i in range(num_classes):
        dets_i = dets.copy()
        bits = classes[i]
        for obs_k, det_id in enumerate(obs_detector_ids_by_obs_index):
            dets_i[:, int(det_id)] = np.uint8(bits[obs_k])

        res = decoder.decode_batch(dets_i)

        if res.decode_time_ms is not None:
            decode_time_total += np.asarray(res.decode_time_ms, dtype=np.float64).reshape(-1)[:num_shots]

        if res.candidate_stage1_costs is None:
            raise ValueError("Decoder did not return candidate_stage1_costs; required for detailed stats.")

        for c_idx, c in enumerate(color_list):
            if c not in res.candidate_costs:
                raise ValueError(f"Decoder result missing candidate_costs[{c!r}]")
            if c not in res.candidate_stage1_costs:
                raise ValueError(f"Decoder result missing candidate_stage1_costs[{c!r}]")
            weights[i, c_idx, :] = np.asarray(res.candidate_costs[c], dtype=np.float64).reshape(-1)[:num_shots]
            s1_weights[i, c_idx, :] = (
                np.asarray(res.candidate_stage1_costs[c], dtype=np.float64).reshape(-1)[:num_shots]
            )

    best_logical_class, _, _, logical_gap_opt = _get_final_predictions(weights)
    if logical_gap_opt is None:
        logical_gap = np.full(num_shots, np.nan, dtype=np.float64)
    else:
        logical_gap = np.asarray(logical_gap_opt, dtype=np.float64).reshape(-1)[:num_shots]

    shot_idx = np.arange(num_shots, dtype=np.int64)
    obs_pred = classes[best_logical_class].astype(np.uint8, copy=False)

    # Fix per-color weights to the selected logical class per shot.
    out_costs: dict[str, np.ndarray] = {}
    out_s1: dict[str, np.ndarray] = {}
    for c_idx, c in enumerate(color_list):
        out_costs[c] = weights[best_logical_class, c_idx, shot_idx]
        out_s1[c] = s1_weights[best_logical_class, c_idx, shot_idx]

    return ComparativeDecodeResult(
        obs_prediction=obs_pred,
        candidate_costs=out_costs,
        candidate_stage1_costs=out_s1,
        decode_time_ms=decode_time_total,
        logical_gap=logical_gap,
    )
