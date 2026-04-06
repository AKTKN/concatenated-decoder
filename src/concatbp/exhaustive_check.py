from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import warnings

import numpy as np

from .circuit_factory import build_circuit
from .config import CircuitConfig, DecoderConfig, ThresholdExperimentConfig
from .dem_model import parse_detector_metadata, parse_stim_dem
from .experiment import _create_decoder, _resolve_task_d2


@dataclass(frozen=True)
class ExhaustiveCheckSummary:
    distance: int
    d2: int | None
    p: float
    basis: str
    noise_model: str
    circuit_style: str
    circuit_from: str
    rounds_spec: str
    rounds_requested: int
    rounds_effective: int
    decoder_strategy: str
    weight: int
    num_detectors: int
    num_observables: int
    num_error_mechanisms: int
    num_patterns: int
    num_fail: int
    num_abort: int


def _comb(n: int, k: int) -> int:
    if k < 0:
        return 0
    if k > n:
        return 0
    return int(math.comb(int(n), int(k)))


def _write_summary_json(out_dir: Path, summary: ExhaustiveCheckSummary) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"exhaustive_check_d={int(summary.distance)}"
        f"_d2={summary.d2 if summary.d2 is not None else 'none'}"
        f"_p={float(summary.p):.12g}"
        f"_basis={str(summary.basis)}"
        f"_w={int(summary.weight)}.json"
    )
    path = out_dir / fname
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_exhaustive_check(
    cfg: ThresholdExperimentConfig,
    decoder_cfg: DecoderConfig,
) -> ExhaustiveCheckSummary:
    """Run an exhaustive low-weight logical check using the naive (monolithic) DEM.

    This mode is intentionally separate from threshold/memory experiments:
    - It runs on a *single* (distance, p) point.
    - It ignores cfg.shots / sampling and instead enumerates all error-mechanism
      patterns of a fixed weight in the naive DEM.

    Success criterion:
        A pattern is counted as failure when the decoded observable flips differ
        from the true observable flips implied by the pattern.

    Notes:
        - If decoder returns an abort mask (post-selection), aborted patterns are
          counted separately and also treated as failures.
    """

    ex_cfg = getattr(cfg, "exhaustive_check", None)
    enabled = bool(getattr(ex_cfg, "enabled", False))
    if not enabled:
        raise ValueError("run_exhaustive_check called but experiment.exhaustive_check.enabled is False")

    weight = int(getattr(ex_cfg, "weight", 1))
    if weight < 0:
        raise ValueError("exhaustive_check.weight must be >= 0")

    # This simulation is intended for a single parameter point.
    distances = list(getattr(cfg, "distances", ()))
    p_vals = list(np.asarray(cfg.resolved_p_values(), dtype=float).reshape(-1))
    if len(distances) != 1 or len(p_vals) != 1:
        raise ValueError(
            "exhaustive_check requires a single point: set experiment.distances to a 1-element list "
            "and set experiment.p_values to a 1-element list (or p_min==p_max and num_p==1). "
            f"Got distances={distances} p_values={p_vals}"
        )

    distance = int(distances[0])
    p = float(p_vals[0])

    if bool(getattr(cfg, "comparative_decoding", False)):
        raise ValueError("exhaustive_check does not support experiment.comparative_decoding")

    # Build circuit/DEM exactly like the main experiment pipeline.
    d2 = _resolve_task_d2(cfg, distance=distance)
    circuit_cfg = CircuitConfig(
        style=cfg.circuit_style,
        diameter=int(distance),
        rounds=cfg.rounds,
        noise_model=cfg.noise_model,  # type: ignore[arg-type]
        circuit_from=cfg.circuit_from,  # type: ignore[arg-type]
        circuit_options=dict(cfg.circuit_options),
        convert_to_cz=bool(getattr(cfg, "convert_to_cz", True)),
        editable_extras=dict(getattr(cfg, "editable_extras", {})),
    )
    built = build_circuit(circuit_cfg, basis=str(cfg.basis), p=p)
    dem = built.circuit.detector_error_model(decompose_errors=False)
    base_model = parse_stim_dem(dem)
    metadata = parse_detector_metadata(built.circuit)

    # Instantiate decoder using the same factory as experiment.py.
    decoder = _create_decoder(
        strategy=decoder_cfg.strategy,
        base_model=base_model,
        dem=dem,
        metadata=metadata,
        decoder_cfg=decoder_cfg,
        debug_base_dir=None,
        debug_run_tag=None,
    )

    num_det = int(base_model.num_detectors)
    num_obs = int(base_model.num_observables)
    num_mech = int(base_model.num_errors)

    num_patterns = _comb(num_mech, weight)

    if weight >= 3:
        warnings.warn(
            f"[concatbp][exhaustive_check] weight={weight} will be very expensive: patterns=C({num_mech},{weight})={num_patterns}",
            RuntimeWarning,
            stacklevel=2,
        )
    elif num_patterns >= 5_000_000:
        warnings.warn(
            f"[concatbp][exhaustive_check] patterns=C({num_mech},{weight})={num_patterns} is very large; this may take a long time.",
            RuntimeWarning,
            stacklevel=2,
        )

    if num_obs == 0:
        warnings.warn(
            "[concatbp][exhaustive_check] Circuit has num_observables=0; logical success/failure cannot be assessed.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Precompute per-mechanism detector/observable flip bit-vectors.
    # This keeps the inner enumeration loop simple and deterministic.
    det_effects = np.zeros((num_mech, num_det), dtype=np.uint8)
    obs_effects = np.zeros((num_mech, num_obs), dtype=np.uint8) if num_obs else np.zeros((num_mech, 0), dtype=np.uint8)

    for i, dets in enumerate(base_model.detector_targets):
        if dets:
            det_effects[i, np.asarray(dets, dtype=np.int64)] ^= 1
    if num_obs:
        for i, obs in enumerate(base_model.observable_targets):
            if obs:
                obs_effects[i, np.asarray(obs, dtype=np.int64)] ^= 1

    # Enumerate patterns and decode in batches.
    batch_size = 1024
    det_batch = np.empty((batch_size, num_det), dtype=np.uint8)
    obs_true_batch = np.empty((batch_size, num_obs), dtype=np.uint8) if num_obs else np.empty((batch_size, 0), dtype=np.uint8)

    num_fail = 0
    num_abort = 0
    processed = 0

    def _decode_and_count(n: int) -> None:
        nonlocal num_fail, num_abort
        if n <= 0:
            return
        res = decoder.decode_batch(det_batch[:n, :])
        obs_pred = np.asarray(res.obs_prediction, dtype=np.uint8)
        if obs_pred.ndim == 1:
            obs_pred = obs_pred.reshape(1, -1)
        obs_pred = obs_pred[:, :num_obs]

        if num_obs:
            mismatch = np.any(obs_pred[:n, :] != obs_true_batch[:n, :], axis=1)
        else:
            mismatch = np.zeros(n, dtype=bool)

        if res.abort_mask is not None:
            abort_flags = np.asarray(res.abort_mask, dtype=bool).reshape(-1)[:n]
            num_abort += int(np.count_nonzero(abort_flags))
            mismatch = mismatch | abort_flags

        num_fail += int(np.count_nonzero(mismatch))

    if weight == 0:
        det_batch[0, :] = 0
        if num_obs:
            obs_true_batch[0, :] = 0
        _decode_and_count(1)
        processed = 1
    elif weight == 1:
        for start in range(0, num_mech, batch_size):
            end = min(num_mech, start + batch_size)
            n = int(end - start)
            det_batch[:n, :] = det_effects[start:end, :]
            if num_obs:
                obs_true_batch[:n, :] = obs_effects[start:end, :]
            _decode_and_count(n)
            processed += n
    else:
        comb_iter = itertools.combinations(range(num_mech), weight)
        k = 0
        for idxs in comb_iter:
            # Initialize the pattern from the first mechanism and XOR the remaining ones.
            det_batch[k, :] = det_effects[idxs[0], :]
            if num_obs:
                obs_true_batch[k, :] = obs_effects[idxs[0], :]
            for j in idxs[1:]:
                det_batch[k, :] ^= det_effects[j, :]
                if num_obs:
                    obs_true_batch[k, :] ^= obs_effects[j, :]

            k += 1
            if k >= batch_size:
                _decode_and_count(k)
                processed += k
                k = 0

        if k:
            _decode_and_count(k)
            processed += k

    if processed != num_patterns:
        # This should not happen, but keep the check to avoid silent partial results.
        raise RuntimeError(
            f"exhaustive_check processed mismatch: processed={processed} expected={num_patterns} "
            f"(num_mech={num_mech}, weight={weight})"
        )

    summary = ExhaustiveCheckSummary(
        distance=int(distance),
        d2=d2,
        p=float(p),
        basis=str(cfg.basis),
        noise_model=str(cfg.noise_model),
        circuit_style=str(built.style),
        circuit_from=str(cfg.circuit_from),
        rounds_spec=str(built.rounds_spec),
        rounds_requested=int(built.rounds_requested),
        rounds_effective=int(built.rounds_effective),
        decoder_strategy=str(decoder_cfg.strategy),
        weight=int(weight),
        num_detectors=int(num_det),
        num_observables=int(num_obs),
        num_error_mechanisms=int(num_mech),
        num_patterns=int(num_patterns),
        num_fail=int(num_fail),
        num_abort=int(num_abort),
    )

    out_dir = Path(cfg.output_dir)
    _write_summary_json(out_dir, summary)
    return summary
