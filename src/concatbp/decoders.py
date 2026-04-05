from __future__ import annotations

import time
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymatching
import scipy.sparse as sp
import stim

from .config import DecoderConfig
from .dem_model import BinaryErrorModel, update_dem_priors_from_posterior_llr
from .partition import (
    StageModels, 
    build_stage_models, 
    split_stage1_dem_by_basis, 
    split_stage2_dem_by_basis, 
    update_basis_dem_priors, 
    _belief_matching_weights,
    llr_to_prob,
    split_base_dem_by_basis,
    _raise_if_hyperedge_mechanisms,
    drop_hyperedge_error_mechanisms_from_stage_models,
    drop_hyperedge_error_mechanisms,
)
from .path_setup import ensure_local_imports


def _import_chromobius_api() -> Any:
    """Imports the chromobius python extension module.

    This repo includes a top-level `chromobius/` directory (C++ sources) that becomes
    a namespace package when importing from the project root, which can shadow the
    actual compiled python extension module.

    To be robust inside this monorepo, we preferentially try to import the compiled
    extension by temporarily prioritizing the `chromobius/` folder containing the
    built `chromobius.cpython-*.so` artifact.
    """
    try:
        import chromobius  # type: ignore
        if hasattr(chromobius, "compile_decoder_for_dem") and hasattr(chromobius, "CompiledDecoder"):
            return chromobius
    except Exception:
        pass

    # Attempt to import the compiled extension from the local chromobius build artifact.
    from pathlib import Path
    import sys

    # src-layout: <repo>/src/concatbp/...
    project_root = Path(__file__).resolve().parents[2]
    chromobius_dir = project_root / "chromobius"
    if chromobius_dir.exists():
        sys.modules.pop("chromobius", None)
        sys.path.insert(0, str(chromobius_dir))
        try:
            import chromobius  # type: ignore
            if hasattr(chromobius, "compile_decoder_for_dem") and hasattr(chromobius, "CompiledDecoder"):
                return chromobius
        finally:
            # Keep sys.path stable: remove only the entry we inserted (if still present at front).
            if sys.path and sys.path[0] == str(chromobius_dir):
                sys.path.pop(0)

    raise RuntimeError(
        "chromobius decoder is not available. "
        "Ensure the chromobius python extension is built/available for the running Python (ABI must match). "
        "In this monorepo, a common approach is building `chromobius/chromobius.cpython-*.so` for your interpreter."
    )


class ChromobiusDecoder:
    """Decoder wrapper using the chromobius library directly on the circuit DEM.

    - Uses the stim DetectorErrorModel produced from the circuit as-is.
    - Does NOT perform two-stage splitting or DEM modification.
    - Post-selection is intentionally not supported (caller should ignore it).
    """

    def __init__(self, dem: stim.DetectorErrorModel):
        self._chromobius = _import_chromobius_api()
        # Compile once per task/circuit.
        if hasattr(self._chromobius, "compile_decoder_for_dem"):
            self._decoder = self._chromobius.compile_decoder_for_dem(dem)
        else:
            self._decoder = self._chromobius.CompiledDecoder.from_dem(dem)

        # Metadata for packing/unpacking.
        self._num_detectors = int(getattr(dem, "num_detectors"))
        self._num_observables = int(getattr(dem, "num_observables"))

    def decode_batch(self, detector_outcomes: np.ndarray) -> DecodeBatchResult:
        dets = np.asarray(detector_outcomes)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        # Convert to bit-packed dets expected by chromobius.
        det_bytes = (self._num_detectors + 7) // 8
        if dets.dtype == np.uint8 and dets.shape[1] == det_bytes:
            dets_packed = dets
        else:
            dets_bool = dets.astype(bool, copy=False)
            dets_packed = np.packbits(dets_bool, axis=1, bitorder="little")
            dets_packed = np.asarray(dets_packed, dtype=np.uint8)

        obs_packed = self._decoder.predict_obs_flips_from_dets_bit_packed(dets_packed)
        obs_packed = np.asarray(obs_packed, dtype=np.uint8)
        if self._num_observables == 0:
            obs = np.zeros((dets.shape[0], 0), dtype=np.uint8)
        else:
            obs_bits = np.unpackbits(obs_packed, axis=1, bitorder="little")
            obs = np.asarray(obs_bits[:, : self._num_observables], dtype=np.uint8)

        # candidate_costs is unused by the experiment pipeline for this decoder.
        return DecodeBatchResult(
            obs_prediction=obs,
            candidate_costs={"chromobius": np.zeros(obs.shape[0], dtype=np.float64)},
            detailed_stats=None,
            abort_mask=None,
        )

    def print_stage1_bp_summary(self, *, tag: str | None = None) -> None:
        return


def _warn_if_hyperedge_drops(
    *,
    cfg: DecoderConfig,
    decoder_label: str,
    basis_label: str,
    drops_by_color: list[tuple[str, int, int]],
) -> None:
    """Optionally warns when hyperedge mechanisms were dropped.

    Args:
        cfg: Decoder config (controls on/off).
        decoder_label: Human-readable decoder context.
        basis_label: Basis tag for the graphlike DEM (e.g. 'z').
        drops_by_color: List of (color, dropped_stage1, dropped_stage2) entries.
    """
    if not bool(getattr(cfg, "warn_on_hyperedge_drops", True)):
        return
    if not drops_by_color:
        return

    parts: list[str] = []
    total = 0
    for color, d1, d2 in drops_by_color:
        if d1 <= 0 and d2 <= 0:
            continue
        total += max(0, int(d1)) + max(0, int(d2))
        stage_bits: list[str] = []
        if d1 > 0:
            stage_bits.append(f"s1={int(d1)}")
        if d2 > 0:
            stage_bits.append(f"s2={int(d2)}")
        parts.append(f"{basis_label}[{color}]:" + ",".join(stage_bits))

    if not parts:
        return

    msg = (
        "[concatbp] Dropped hyperedge mechanisms while building graphlike DEMs "
        "(ignore_hyperedge_error_mechanism=True): "
        f"{decoder_label} total={total} "
        + "; ".join(parts)
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=2)

# Optional local-import bootstrap:
# - When installed as a proper package, we should NOT mutate sys.path.
# - When running from this monorepo, we *may* need to add sibling src paths.
#
# We therefore attempt normal imports first, and only fall back to ensure_local_imports
# when imports fail.

_IMPORT_ERROR_LDPC: Exception | None = None
_IMPORT_ERROR_RELAY: Exception | None = None

try:
    from ldpc.bplsd_decoder import BpLsdDecoder  # type: ignore
    from ldpc.union_find_decoder import UnionFindDecoder  # type: ignore
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR_LDPC = exc
    try:
        ensure_local_imports()
        from ldpc.bplsd_decoder import BpLsdDecoder  # type: ignore
        from ldpc.union_find_decoder import UnionFindDecoder  # type: ignore
        _IMPORT_ERROR_LDPC = None
    except Exception as exc2:
        _IMPORT_ERROR_LDPC = exc2
        BpLsdDecoder = None  # type: ignore[assignment]
        UnionFindDecoder = None  # type: ignore[assignment]

try:
    import relay_bp  # type: ignore
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR_RELAY = exc
    try:
        ensure_local_imports()
        import relay_bp  # type: ignore
        _IMPORT_ERROR_RELAY = None
    except Exception as exc2:
        _IMPORT_ERROR_RELAY = exc2
        relay_bp = None  # type: ignore[assignment]

def _safe_log_weights(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return np.log((1 - p) / p)


@dataclass(frozen=True)
class DecodeBatchResult:
    """Batch decode result.

    This object is intentionally a simple data container so experiment runners can
    optionally persist additional per-shot/per-candidate diagnostics (e.g. stage-1
    costs, per-shot decode time) without re-decoding.
    """
    obs_prediction: np.ndarray
    candidate_costs: dict[str, np.ndarray]
    # Optional per-candidate stage-1 weights/costs. Keys match candidate_costs.
    candidate_stage1_costs: dict[str, np.ndarray] | None = None
    # Optional candidate observable predictions per color. Shape per entry: (shots, num_obs).
    candidate_obs: dict[str, np.ndarray] | None = None
    # Optional per-shot total decoding time (ms). Includes all evaluated colors.
    decode_time_ms: np.ndarray | None = None
    # Optional per-shot logical gap (comparative decoding only).
    logical_gap: np.ndarray | None = None
    # Optional per-shot iteration count (relay_bp strategy).
    iterations: np.ndarray | None = None
    detailed_stats: list[dict[str, int | float | str]] | None = None
    abort_mask: np.ndarray | None = None

class _BPLSDStageDecoder:
    """Wrapper for the standard BPLSD decoder."""
    def __init__(self, model: BinaryErrorModel, cfg: DecoderConfig):
        self.model = model
        self.cfg = cfg
        self.weights = _safe_log_weights(model.probabilities)
        self.h, self.o = model.to_matrices()
        if model.num_errors > 0 and model.num_detectors > 0:
            if BpLsdDecoder is None:
                raise RuntimeError(
                    "BpLsdDecoder is not available. Install 'ldpc' (or ensure local src paths are present). "
                    f"Original import error: {_IMPORT_ERROR_LDPC}"
                )
            self.decoder = BpLsdDecoder(
                self.h,
                error_channel=list(np.clip(model.probabilities, 1e-12, 1.0 - 1e-12)),
                max_iter=cfg.max_iter,
                bp_method=cfg.bp_method,
                schedule=cfg.schedule,
                ms_scaling_factor=cfg.ms_scaling_factor,
                lsd_method=cfg.lsd_method,
                lsd_order=cfg.lsd_order,
            )
        else:
            self.decoder = None

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if self.model.num_errors == 0:
            return np.zeros(0, dtype=np.uint8), np.zeros(self.model.num_observables, dtype=np.uint8), 0.0

        if self.decoder is None:
            e = np.zeros(self.model.num_errors, dtype=np.uint8)
        else:
            e = np.asarray(self.decoder.decode(syndrome.astype(np.uint8)), dtype=np.uint8)
        obs = np.asarray((self.o @ e) % 2, dtype=np.uint8).ravel()
        cost = float(np.dot(self.weights, e))
        return e, obs, cost
    

@dataclass(frozen=True)
class _RelayDetailedResult:
    error_prediction: np.ndarray
    observables_prediction: np.ndarray
    cost: float
    converged: bool
    iterations: int
    max_iter: int
    postrior_llr: np.ndarray | None = None

class _RelayBPStageDecoder:
    """Wrapper for the custom Relay BP decoder."""
    def __init__(self, model: BinaryErrorModel, cfg: DecoderConfig):
        self.model = model
        self.cfg = cfg
        self.weights = _safe_log_weights(model.probabilities)
        self.h, self.o = model.to_matrices()
        self.max_iter = int(cfg.relay_bp.pre_iter + cfg.relay_bp.num_sets * cfg.relay_bp.set_max_iter)

        if relay_bp is None:
            raise RuntimeError(
                "relay_bp is not available. Install/build 'relay_bp' (or ensure local src paths are present). "
                f"Original import error: {_IMPORT_ERROR_RELAY}"
            )

        if model.num_errors > 0 and model.num_detectors > 0:
            variant = str(cfg.relay_bp.decoder_variant).lower()
            decoder_cls_map = {
                "f32": relay_bp.RelayDecoderF32,
                "f64": relay_bp.RelayDecoderF64,
                "i32": relay_bp.RelayDecoderI32,
                "i64": relay_bp.RelayDecoderI64,
            }
            interval = tuple(float(v) for v in cfg.relay_bp.gamma_dist_interval)
            self.decoder = decoder_cls_map[variant](
                self.h,
                error_priors=np.clip(model.probabilities, 1e-12, 1 - 1e-12),
                gamma0=float(cfg.relay_bp.gamma0),
                pre_iter=int(cfg.relay_bp.pre_iter),
                num_sets=int(cfg.relay_bp.num_sets),
                set_max_iter=int(cfg.relay_bp.set_max_iter),
                gamma_dist_interval=interval,
                stop_nconv=int(cfg.relay_bp.stop_nconv),
            )
        else:
            self.decoder = None

    def decode_detailed(self, syndrome: np.ndarray) -> _RelayDetailedResult:
        if self.model.num_errors == 0 or self.decoder is None:
            e = np.zeros(self.model.num_errors, dtype=np.uint8)
            obs = np.asarray((self.o @ e) % 2, dtype=np.uint8).ravel()
            return _RelayDetailedResult(e, obs, 0.0, True, 0, self.max_iter)

        res = self.decoder.decode_detailed(syndrome.astype(np.uint8))
        e = np.asarray(res.decoding, dtype=np.uint8)
        obs = np.asarray((self.o @ e) % 2, dtype=np.uint8).ravel()
        cost = float(np.dot(self.weights, e))
        return _RelayDetailedResult(
            error_prediction=e,
            observables_prediction=obs,
            cost=cost,
            converged=bool(getattr(res, "converged", getattr(res, "success", False))),
            iterations=int(getattr(res, "iterations", self.max_iter)),
            max_iter=int(getattr(res, "max_iter", self.max_iter)),
            postrior_llr=np.asarray(getattr(res, "posterior_ratios", None)) if hasattr(res, "posterior_ratios") else None
        )
    
class _MatchingStageDecoder:
    """Wrapper for MWPM or UF decoder, supporting dynamic weight updates."""
    def __init__(self, model: BinaryErrorModel, matcher_type: str):
        self.model = model
        self.matcher_type = matcher_type.lower()
        self.weights = _safe_log_weights(model.probabilities)
        self.h, self.o = model.to_matrices()
        
        if model.num_errors > 0 and model.num_detectors > 0:
            if self.matcher_type == "uf":
                # ldpc.union_find_decoder.UnionFindDecoder accepts only (pcm, uf_method=False).
                # Soft information (priors) are passed at decode time via llrs.
                if UnionFindDecoder is None:
                    raise RuntimeError(
                        "UnionFindDecoder is not available. Install 'ldpc' (or ensure local src paths are present). "
                        f"Original import error: {_IMPORT_ERROR_LDPC}"
                    )
                self.decoder = UnionFindDecoder(self.h)
            else:
                self.decoder = pymatching.Matching(self.h, error_weights=self.weights)
        else:
            self.decoder = None
    
    def decode(self, syndrome: np.ndarray, custom_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        if self.model.num_errors == 0:
            return np.zeros(0, dtype=np.uint8), np.zeros(self.model.num_observables, dtype=np.uint8), 0.0
        
        if self.decoder is None:
            e = np.zeros(self.model.num_errors, dtype=np.uint8)
            cost = 0.0
        else:
            if self.matcher_type == "uf":
                if custom_weights is not None:
                    e = np.asarray(self.decoder.decode(syndrome.astype(np.uint8), llrs=custom_weights), dtype=np.uint8)
                    cost = float(np.dot(custom_weights, e))
                else:
                    # Use model-derived priors by default.
                    e = np.asarray(self.decoder.decode(syndrome.astype(np.uint8), llrs=self.weights), dtype=np.uint8)
                    cost = float(np.dot(self.weights, e))
            else:
                if custom_weights is not None:
                    # Dynamically re-instantiate matching graph for updated BP priors
                    temp_decoder = pymatching.Matching(self.h, error_weights=custom_weights)
                    e = np.asarray(temp_decoder.decode(syndrome.astype(bool)), dtype=np.uint8)
                    cost = float(np.dot(custom_weights, e))
                else:
                    e = np.asarray(self.decoder.decode(syndrome.astype(bool)), dtype=np.uint8)
                    cost = float(np.dot(self.weights, e))
                    
        obs = np.asarray((self.o @ e) % 2, dtype=np.uint8).ravel()
        return e, obs, cost

class _Stage1GraphDecoder:
    """Handles X/Z basis separation and matching for Stage 1."""
    def __init__(self, model: BinaryErrorModel, input_dets: list[int], basis_by_id: dict[int, str], matcher_type: str, ignore_hyperedges: bool = True):
        # X Basis (Detects Z errors)
        raw_x_model, raw_x_mech_map, self.x_det_map = split_stage1_dem_by_basis(model, input_dets, basis_by_id, "X")
        if ignore_hyperedges:
            self.x_model, self.x_mech_map, _ = drop_hyperedge_error_mechanisms(raw_x_model, raw_x_mech_map)
        else:
            self.x_model, self.x_mech_map = raw_x_model, raw_x_mech_map
        self.x_matcher = _MatchingStageDecoder(self.x_model, matcher_type)

        # Z Basis (Detects X errors)
        raw_z_model, raw_z_mech_map, self.z_det_map = split_stage1_dem_by_basis(model, input_dets, basis_by_id, "Z")
        if ignore_hyperedges:
            self.z_model, self.z_mech_map, _ = drop_hyperedge_error_mechanisms(raw_z_model, raw_z_mech_map)
        else:
            self.z_model, self.z_mech_map = raw_z_model, raw_z_mech_map
        self.z_matcher = _MatchingStageDecoder(self.z_model, matcher_type)

class _Stage2GraphDecoder:
    """Handles X/Z basis separation and matching for Stage 2."""
    def __init__(
        self, s2_model: BinaryErrorModel, s2_real_dets: list[int],
        s1_model: BinaryErrorModel, s1_input_dets: list[int],
        basis_by_id: dict[int, str], matcher_type: str, keep_obs_on: str = "Z",
        ignore_hyperedges: bool = True
    ):
        # X Basis (Detects Z errors)
        raw_x_model, raw_x_mech_map, self.x_det_map = split_stage2_dem_by_basis(
            s2_model, s2_real_dets, s1_model, s1_input_dets, basis_by_id, "X", keep_observables=(keep_obs_on == "X")
        )
        if ignore_hyperedges:
            self.x_model, self.x_mech_map, _ = drop_hyperedge_error_mechanisms(raw_x_model, raw_x_mech_map)
        else:
            self.x_model, self.x_mech_map = raw_x_model, raw_x_mech_map
        self.x_matcher = _MatchingStageDecoder(self.x_model, matcher_type)

        # Z Basis (Detects X errors)
        raw_z_model, raw_z_mech_map, self.z_det_map = split_stage2_dem_by_basis(
            s2_model, s2_real_dets, s1_model, s1_input_dets, basis_by_id, "Z", keep_observables=(keep_obs_on == "Z")
        )
        if ignore_hyperedges:
            self.z_model, self.z_mech_map, _ = drop_hyperedge_error_mechanisms(raw_z_model, raw_z_mech_map)
        else:
            self.z_model, self.z_mech_map = raw_z_model, raw_z_mech_map
        self.z_matcher = _MatchingStageDecoder(self.z_model, matcher_type)


class BaseConcatDecoder(ABC):
    """Abstract base class for two-stage concatenated decoders.

    Responsibilities:
    - Partitions the DEM into per-color two-stage models.
    - Implements unique-syndrome caching across shots.
    - Aggregates optional debug metrics in a result-stable way.
    """
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        self.base_model = base_model
        self.cfg = cfg
        self.detector_basis_by_id = kwargs.get("detector_basis_by_id", {})
        # Debug counters (primarily for belief_* strategies that expose stage1_bp_failed).
        colors = list(cfg.colors)
        self._debug_stage1_bp_attempts_by_color: dict[str, int] = {str(c): 0 for c in colors}
        self._debug_stage1_bp_failed_by_color: dict[str, int] = {str(c): 0 for c in colors}
        # "Selected" means the color that achieved the lowest cost for the (unique) syndrome.
        self._debug_stage1_bp_selected_attempts: int = 0
        self._debug_stage1_bp_selected_failed: int = 0
        
        # Build global partitioned models
        self.stage_models: dict[str, StageModels] = {
            c: build_stage_models(base_model, detector_ids_by_color[c]) for c in cfg.colors
        }
    
    @abstractmethod
    def _decode_single_syndrome(
        self,
        c: str,
        syn_global: np.ndarray,
        s1_cache: dict,
        s2_cache: dict,
    ) -> tuple[np.ndarray, float, dict[str, int | float | str]]:
        """Decode one unique syndrome under a given color hypothesis.

        Returns:
            obs_pred: Predicted observables (uint8 vector).
            stage2_cost: Stage-2 (and/or total) cost used for cross-color selection.
            stats: Optional metrics (e.g. decode_time_ms, stage1_weight).
        """
        pass
    
    def decode_batch(self, detector_outcomes: np.ndarray) -> DecodeBatchResult:
        dets = np.asarray(detector_outcomes, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        
        num_shots = dets.shape[0]
        unique_dets, inverse = np.unique(dets, axis=0, return_inverse=True)
        num_unique = unique_dets.shape[0]
        counts_per_unique = np.bincount(inverse, minlength=num_unique).astype(np.int64)
        
        best_obs_unique = np.zeros((num_unique, self.base_model.num_observables), dtype=np.uint8)
        best_cost_unique = np.full(num_unique, np.inf, dtype=np.float64)
        # Resolve which color hypotheses to evaluate.
        css = getattr(self.cfg, "color_selection_strategy", "all")
        if isinstance(css, str):
            if css != "all":
                raise ValueError("color_selection_strategy must be 'all' or a list of colors")
            colors = [str(c) for c in self.cfg.colors]
        else:
            colors = [str(c) for c in css]

        color_to_index = {c: i for i, c in enumerate(colors)}
        best_color_unique = np.full(num_unique, -1, dtype=np.int16)
        all_costs_unique = {c: np.zeros(num_unique, dtype=np.float64) for c in colors}

        # Optional per-color stage-1 weights.
        all_stage1_costs_unique = {c: np.full(num_unique, np.nan, dtype=np.float64) for c in colors}

        # Total decode time per unique syndrome (ms), aggregated across evaluated colors.
        decode_time_unique_ms = np.zeros(num_unique, dtype=np.float64)
        # Optional selected-candidate iteration count per unique syndrome.
        iterations_unique: np.ndarray | None = None

        need_postselect = bool(getattr(self.cfg, "do_post_selection", False)) and len(colors) >= 2

        # Always track per-color observable predictions. This is small (num_obs is small)
        # and enables shot-level stats/signing without re-decoding.
        candidate_obs_unique: dict[str, np.ndarray] = {
            c: np.zeros((num_unique, self.base_model.num_observables), dtype=np.uint8)
            for c in colors
        }

        # Optional debug collection for belief_* strategies.
        stage1_bp_failed_unique: dict[str, np.ndarray] = {c: np.zeros(num_unique, dtype=np.uint8) for c in colors}
        stage1_bp_stats_seen = False
        
        for c in colors:
            s1_cache: dict[bytes, tuple[np.ndarray, np.ndarray, float]] = {}
            s2_cache: dict[bytes, tuple[np.ndarray, np.ndarray, float]] = {}

            for i in range(num_unique):
                syn_global = unique_dets[i]
                
                # Dynamic dispatch to subclass strategy
                obs2, total_cost, stats = self._decode_single_syndrome(c, syn_global, s1_cache, s2_cache)
                candidate_obs_unique[str(c)][i] = np.asarray(obs2, dtype=np.uint8).ravel()

                if "stage1_weight" in stats:
                    all_stage1_costs_unique[str(c)][i] = float(stats["stage1_weight"])  # type: ignore[arg-type]

                if "decode_time_ms" in stats:
                    decode_time_unique_ms[i] += float(stats["decode_time_ms"])  # type: ignore[arg-type]

                if "stage1_bp_failed" in stats:
                    stage1_bp_stats_seen = True
                    stage1_bp_failed_unique[str(c)][i] = 1 if int(stats["stage1_bp_failed"]) else 0

                shot_iteration: float | None = None
                if "iteration" in stats:
                    if iterations_unique is None:
                        iterations_unique = np.full(num_unique, np.nan, dtype=np.float64)
                    shot_iteration = float(stats["iteration"])  # type: ignore[arg-type]

                all_costs_unique[str(c)][i] = total_cost
                if total_cost < best_cost_unique[i]:
                    best_cost_unique[i] = total_cost
                    best_obs_unique[i] = obs2
                    best_color_unique[i] = int(color_to_index[str(c)])
                    if iterations_unique is not None and shot_iteration is not None:
                        iterations_unique[i] = shot_iteration

        # Aggregate Stage-1 BP non-convergence statistics weighted by shot multiplicity.
        if stage1_bp_stats_seen:
            total_shots = int(num_shots)
            for c in colors:
                self._debug_stage1_bp_attempts_by_color[c] += total_shots
                self._debug_stage1_bp_failed_by_color[c] += int(np.dot(stage1_bp_failed_unique[c].astype(np.int64), counts_per_unique))

            self._debug_stage1_bp_selected_attempts += total_shots
            # Which color was selected for each unique syndrome?
            selected_failed_flags = np.zeros(num_unique, dtype=np.int64)
            for i in range(num_unique):
                idx = int(best_color_unique[i])
                if idx < 0:
                    continue
                selected_color = colors[idx]
                selected_failed_flags[i] = int(stage1_bp_failed_unique[selected_color][i])
            self._debug_stage1_bp_selected_failed += int(np.dot(selected_failed_flags, counts_per_unique))

        best_obs = best_obs_unique[inverse]
        all_costs = {c: all_costs_unique[c][inverse] for c in colors}

        abort_mask: np.ndarray | None = None
        if need_postselect:
            # Abort when candidate solutions disagree on observables.
            ref = candidate_obs_unique[colors[0]]
            abort_unique = np.zeros(num_unique, dtype=bool)
            for c in colors[1:]:
                abort_unique |= np.any(candidate_obs_unique[c] != ref, axis=1)
            abort_mask = abort_unique[inverse]

        candidate_obs = {c: candidate_obs_unique[c][inverse] for c in colors}
        candidate_stage1_costs = {c: all_stage1_costs_unique[c][inverse] for c in colors}
        decode_time_ms = decode_time_unique_ms[inverse]
        iterations = iterations_unique[inverse] if iterations_unique is not None else None

        return DecodeBatchResult(
            obs_prediction=best_obs,
            candidate_costs=all_costs,
            candidate_stage1_costs=candidate_stage1_costs,
            candidate_obs=candidate_obs,
            decode_time_ms=decode_time_ms,
            iterations=iterations,
            detailed_stats=None,
            abort_mask=abort_mask,
        )

    def print_stage1_bp_summary(self, *, tag: str | None = None) -> None:
        """Print accumulated Stage-1 BP non-convergence statistics.

        Intended for belief_* strategies. Safe to call for any decoder.
        """
        if not bool(getattr(self.cfg, "debug_print_stage1_bp_summary", False)):
            return

        attempted_total = sum(int(v) for v in self._debug_stage1_bp_attempts_by_color.values())
        if attempted_total <= 0 and self._debug_stage1_bp_selected_attempts <= 0:
            return

        parts: list[str] = []
        if tag:
            parts.append(f"tag={tag}")

        per_color = {
            c: {
                "attempted": int(self._debug_stage1_bp_attempts_by_color.get(c, 0)),
                "failed": int(self._debug_stage1_bp_failed_by_color.get(c, 0)),
            }
            for c in sorted(self._debug_stage1_bp_attempts_by_color.keys())
        }
        selected_attempted = int(self._debug_stage1_bp_selected_attempts)
        selected_failed = int(self._debug_stage1_bp_selected_failed)
        selected_rate = (selected_failed / selected_attempted) if selected_attempted else float("nan")
        parts.append(f"selected_failed={selected_failed}/{selected_attempted} ({selected_rate:.6g})")
        parts.append(f"per_color={per_color}")
        print(f"[concatbp][stage1_bp_nonconvergence_summary] {{" + ", ".join(parts) + "}}")
    


# =====================================================================
# Specific Decoder Strategy Implementations
# =====================================================================

class ConcatBPLSDDecoder(BaseConcatDecoder):
    """Standard Two-Stage Decoder using BPLSD natively for both stages."""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, **kwargs)
        self.stage1_decoders = {c: _BPLSDStageDecoder(m.stage1, cfg) for c, m in self.stage_models.items()}
        self.stage2_decoders = {c: _BPLSDStageDecoder(m.stage2, cfg) for c, m in self.stage_models.items()}

    def _decode_single_syndrome(self, c: str, syn_global: np.ndarray, s1_cache: dict, s2_cache: dict) -> tuple[np.ndarray, float, dict]:
        models = self.stage_models[c]
        t0 = time.perf_counter()
        
        syn1 = syn_global[models.stage1_input_detectors]
        e1, _, cost1 = self.stage1_decoders[c].decode(syn1) # Cache logic can be injected here
        
        syn2_real = syn_global[models.stage2_real_detectors]
        syn2 = np.concatenate([syn2_real, e1.astype(np.uint8)], axis=0)
        
        _, obs2, cost2 = self.stage2_decoders[c].decode(syn2)
        
        return obs2, cost2, {
            "stage1_weight": float(cost1),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }

class ConcatRelayBPDecoder(BaseConcatDecoder):
    """
    Standard Two-Stage Decoder using Relay BP natively for both stages.
    Does not perform any dynamic fallback or hybrid graph matching.
    """
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, **kwargs)
        
        if relay_bp is None:
            raise RuntimeError("relay_bp module is not available, but ConcatRelayBPDecoder was requested.")
            
        self.stage1_decoders = {c: _RelayBPStageDecoder(m.stage1, cfg) for c, m in self.stage_models.items()}
        self.stage2_decoders = {c: _RelayBPStageDecoder(m.stage2, cfg) for c, m in self.stage_models.items()}

    def _decode_single_syndrome(self, c: str, syn_global: np.ndarray, s1_cache: dict, s2_cache: dict) -> tuple[np.ndarray, float, dict]:
        models = self.stage_models[c]
        t0 = time.perf_counter()
    
        syn1 = syn_global[models.stage1_input_detectors]
        
        # Note: s1_cache logic could be added here if caching Relay-BP results is desired.
        res1 = self.stage1_decoders[c].decode_detailed(syn1)
        e1_virtual = res1.error_prediction
        
        syn2_real = syn_global[models.stage2_real_detectors]
        syn2 = np.concatenate([syn2_real, e1_virtual.astype(np.uint8)], axis=0)
        
        res2 = self.stage2_decoders[c].decode_detailed(syn2)
        
        # Since Relay BP returns detailed stats, it's useful to log its convergence status
        stats = {
            "stage1_converged": int(res1.converged),
            "stage2_converged": int(res2.converged),
            "stage1_weight": float(res1.cost),
            "iteration": int(res1.iterations) + int(res2.iterations),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }
        
        return res2.observables_prediction, res2.cost, stats


class ConcatBeliefGraphDecoder(BaseConcatDecoder):
    """
    Core implementation for Belief-Matching/Find hybrid strategy.
    Runs BP on the full Stage 1 model. 
    [Converged]: Projects the hard decision down to the Z-basis graph using XOR parity.
    [Failed]: Projects the BP posteriors down to the Z-basis graph using Sum marginalization.
    Both paths then seamlessly hand off to the Pure Concat MWPM/UF pipeline.
    """
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, matcher_type: str, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, **kwargs)
        if not self.detector_basis_by_id:
            raise ValueError("detector_basis_by_id must be provided for Belief-Graph hybrid strategies.")

        # Remember the graph-decoder type (mwpm or uf).
        self.matcher_type = matcher_type.lower()
        
        self._bp_posterior_debug_prints = 0
        self._bp_failure_prior_debug_prints = 0
        
        # 1. Stage 1 BP Init (Full Model)
        self.stage1_bps = {c: _RelayBPStageDecoder(m.stage1, cfg) for c, m in self.stage_models.items()}
        
        target_obs_basis = getattr(cfg.hybrid_two_stage, "stage2_graph_target_basis", "Z")
        ignore_hyperedges = bool(getattr(cfg, "ignore_hyperedge_error_mechanism", True))

        # 2. Pure Z-Graph Pipeline Init (Exact replica of ConcatGraphDecoder)
        self.base_z, self.z_global_map, self.z_base_mech_map = split_base_dem_by_basis(
            base_model, self.detector_basis_by_id, "Z", keep_observables=(target_obs_basis in ["Z", "auto"])
        )
        z_global_to_local = {g_id: l_id for l_id, g_id in enumerate(self.z_global_map)}
        
        self.fallback_z_stages = {}
        self.fallback_z_matchers = {}
        self.s1_to_z_s1_map = {}  # Projection mapping: Z-graph idx -> list of full-model Stage-1 idxs.

        hyperedge_drops: list[tuple[str, int, int]] = []

        for c in cfg.colors:
            color_dets_global = detector_ids_by_color[c]
            color_dets_z = [z_global_to_local[d] for d in color_dets_global if d in z_global_to_local]
            
            z_models = build_stage_models(self.base_z, color_dets_z)
            if ignore_hyperedges:
                pre_s1 = int(z_models.stage1.num_errors)
                pre_s2 = int(z_models.stage2.num_errors)
                z_models_dropped = drop_hyperedge_error_mechanisms_from_stage_models(z_models)
                d1 = pre_s1 - int(z_models_dropped.stage1.num_errors)
                d2 = pre_s2 - int(z_models_dropped.stage2.num_errors)
                if d1 > 0 or d2 > 0:
                    hyperedge_drops.append((str(c), int(d1), int(d2)))
                z_models = z_models_dropped
            
            self.fallback_z_stages[c] = z_models
            self.fallback_z_matchers[c] = {
                "stage1": _MatchingStageDecoder(z_models.stage1, self.matcher_type),
                "stage2": _MatchingStageDecoder(z_models.stage2, self.matcher_type)
            }

            # ========================================================
            # 3) Build projection mapping (Full Stage 1 -> Pure-Z Stage 1)
            # ========================================================
            # a) Reverse lookup: global_idx -> z_s1_idx
            base_to_z_s1 = {}
            for z_s1_idx, z_base_dict in enumerate(z_models.stage1_mech_mapping):
                for z_base_idx in z_base_dict.keys():
                    for global_idx in self.z_base_mech_map[z_base_idx]:
                        base_to_z_s1[global_idx] = z_s1_idx

            # b) Forward map: s1_idx -> global_idx -> z_s1_idx
            z_s1_mech_map = [[] for _ in range(z_models.stage1.num_errors)]
            for s1_idx, base_dict in enumerate(self.stage_models[c].stage1_mech_mapping):
                global_idx = next(iter(base_dict.keys()))
                if global_idx in base_to_z_s1:
                    z_s1_idx = base_to_z_s1[global_idx]
                    z_s1_mech_map[z_s1_idx].append(s1_idx)
            
            self.s1_to_z_s1_map[c] = z_s1_mech_map

        if ignore_hyperedges:
            _warn_if_hyperedge_drops(
                cfg=cfg,
                decoder_label=self.__class__.__name__,
                basis_label="z",
                drops_by_color=hyperedge_drops,
            )

    def _decode_single_syndrome(self, c: str, syn_global: np.ndarray, s1_cache: dict, s2_cache: dict) -> tuple[np.ndarray, float, dict]:
        models = self.stage_models[c]
        z_models = self.fallback_z_stages[c]
        z_matchers = self.fallback_z_matchers[c]
        s1_to_z = self.s1_to_z_s1_map[c]

        stats = {"stage1_bp_failed": 0, "decode_time_ms": 0.0, "stage1_weight": float("nan")}
        t0 = time.perf_counter()

        # ---------------------------------------------------------
        # STAGE 1: BP Execution
        # ---------------------------------------------------------
        syn1 = syn_global[models.stage1_input_detectors]
        bp_res = self.stage1_bps[c].decode_detailed(syn1)
        
        syn_base_z = syn_global[self.z_global_map]

        if bp_res.converged:
            # Converged: project hard decision via XOR parity.
            e1_full = bp_res.error_prediction
            
            e1_z = np.zeros(z_models.stage1.num_errors, dtype=np.uint8)
            for z_s1_idx, s1_indices in enumerate(s1_to_z):
                parity = 0
                for s1_idx in s1_indices:
                    parity ^= int(e1_full[s1_idx])
                e1_z[z_s1_idx] = parity
                
            stats["stage1_weight"] = float(getattr(bp_res, "cost", float("nan")))
        else:
            # Not converged: project BP posteriors via Sum marginalization, then fallback.
            stats["stage1_bp_failed"] = 1
            bp_llr = getattr(bp_res, "postrior_llr", getattr(bp_res, "posterior_llr", None))
            bp_probs = llr_to_prob(bp_llr)

            # Marginalize BP posteriors onto the Z-graph (additive rule).
            z_priors = np.zeros(z_models.stage1.num_errors, dtype=np.float64)
            for z_s1_idx, s1_indices in enumerate(s1_to_z):
                p_sum = sum(float(bp_probs[s1_idx]) for s1_idx in s1_indices)
                z_priors[z_s1_idx] = min(1.0, p_sum)
            
            # Select which "soft information" weights to pass (UF vs MWPM).
            if self.matcher_type == "uf":
                stage1_weights = _safe_log_weights(z_priors)       # UF uses LLR: log((1-p)/p)
            else:
                stage1_weights = _belief_matching_weights(z_priors) # MWPM uses cost: -log(p)

            syn1_z = syn_base_z[z_models.stage1_input_detectors]
            e1_z, _, cost1_z = z_matchers["stage1"].decode(
                syn1_z, custom_weights=stage1_weights
            )
            stats["stage1_weight"] = float(cost1_z)

        # ---------------------------------------------------------
        # STAGE 2: Pure MWPM/UF Pipeline Execution
        # ---------------------------------------------------------
        syn2_real_z = syn_base_z[z_models.stage2_real_detectors]
        syn2_z = np.concatenate([syn2_real_z, e1_z], axis=0)

        _, obs_z, cost_z = z_matchers["stage2"].decode(syn2_z)

        stats["decode_time_ms"] = (time.perf_counter() - t0) * 1000.0
        return obs_z, cost_z, stats



class ConcatBeliefMatchingDecoder(ConcatBeliefGraphDecoder):
    """Belief-Matching Hybrid: BP -> MWPM"""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="mwpm", **kwargs)


class ConcatBeliefFindDecoder(ConcatBeliefGraphDecoder):
    """Belief-Find Hybrid: BP -> Union-Find"""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="uf", **kwargs)


class ConcatBeliefConcatMWPMDecoder:
    """Run full-DEM Relay-BP first, then fallback to Concat MWPM if BP did not converge."""

    def __init__(
        self,
        base_model: BinaryErrorModel,
        detector_ids_by_color: dict[str, list[int]],
        cfg: DecoderConfig,
        **kwargs,
    ):
        self.base_model = base_model
        self.detector_ids_by_color = detector_ids_by_color
        self.cfg = cfg
        self.detector_basis_by_id: dict[int, str] = kwargs.get("detector_basis_by_id", {})
        if not self.detector_basis_by_id:
            raise ValueError("detector_basis_by_id must be provided for belief_concatmwpm.")

        self._bp_only = bool(getattr(getattr(cfg, "belief_concatmwpm", None), "bp_only", False))
        self._full_bp = _RelayBPStageDecoder(base_model, cfg)
        self._debug_attempted_shots = 0
        self._debug_failed_shots = 0

    def decode_batch(self, detector_outcomes: np.ndarray) -> DecodeBatchResult:
        dets = np.asarray(detector_outcomes, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        num_shots = int(dets.shape[0])
        unique_dets, inverse = np.unique(dets, axis=0, return_inverse=True)
        num_unique = int(unique_dets.shape[0])
        counts_per_unique = np.bincount(inverse, minlength=num_unique).astype(np.int64)

        best_obs_unique = np.zeros((num_unique, self.base_model.num_observables), dtype=np.uint8)
        best_cost_unique = np.full(num_unique, np.inf, dtype=np.float64)
        iterations_unique = np.full(num_unique, np.nan, dtype=np.float64)
        bp_failed_unique = np.zeros(num_unique, dtype=np.uint8)
        abort_unique = np.zeros(num_unique, dtype=bool)

        for i in range(num_unique):
            syn_global = unique_dets[i]
            bp_res = self._full_bp.decode_detailed(syn_global)
            iterations_unique[i] = float(bp_res.iterations)

            if self._bp_only or bp_res.converged:
                best_obs_unique[i] = np.asarray(bp_res.observables_prediction, dtype=np.uint8).ravel()
                best_cost_unique[i] = float(bp_res.cost)
                continue

            bp_failed_unique[i] = 1
            posterior_llr = getattr(bp_res, "postrior_llr", getattr(bp_res, "posterior_llr", None))
            if posterior_llr is None:
                updated_model = self.base_model
            else:
                updated_model = update_dem_priors_from_posterior_llr(self.base_model, posterior_llr)

            concat_decoder = ConcatMatchingDecoder(
                updated_model,
                self.detector_ids_by_color,
                self.cfg,
                detector_basis_by_id=self.detector_basis_by_id,
            )
            concat_res = concat_decoder.decode_batch(syn_global.reshape(1, -1))
            best_obs_unique[i] = np.asarray(concat_res.obs_prediction[0], dtype=np.uint8).ravel()

            if concat_res.abort_mask is not None:
                abort_unique[i] = bool(np.asarray(concat_res.abort_mask).reshape(-1)[0])

            if concat_res.candidate_costs:
                min_cost = min(float(np.asarray(v, dtype=np.float64)[0]) for v in concat_res.candidate_costs.values())
            else:
                min_cost = float("inf")
            best_cost_unique[i] = min_cost

        self._debug_attempted_shots += num_shots
        self._debug_failed_shots += int(np.dot(bp_failed_unique.astype(np.int64), counts_per_unique))

        best_obs = best_obs_unique[inverse]
        candidate_costs = {"belief_concatmwpm": best_cost_unique[inverse]}
        abort_mask = abort_unique[inverse] if bool(getattr(self.cfg, "do_post_selection", False)) else None
        return DecodeBatchResult(
            obs_prediction=best_obs,
            candidate_costs=candidate_costs,
            iterations=iterations_unique[inverse],
            detailed_stats=None,
            abort_mask=abort_mask,
        )

    def print_stage1_bp_summary(self, *, tag: str | None = None) -> None:
        if not bool(getattr(self.cfg, "debug_print_stage1_bp_summary", False)):
            return
        attempted = int(self._debug_attempted_shots)
        if attempted <= 0:
            return
        failed = int(self._debug_failed_shots)
        rate = failed / attempted
        prefix = f"tag={tag} " if tag else ""
        print(
            "[concatbp][belief_concatmwpm_bp_summary] "
            f"{prefix}failed={failed}/{attempted} ({rate:.6g})"
        )


class ConcatGraphDecoder(BaseConcatDecoder):
    """
    Pure graph-based Two-Stage Decoder (No BP). 
    Pre-splits the Base Model into X and Z models, processing them completely independently.
    """
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, matcher_type: str, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, **kwargs)
        if not self.detector_basis_by_id:
            raise ValueError("detector_basis_by_id must be provided for pure Graph decoders.")

        detector_color_by_global: dict[int, str] = {}
        for color_name, ids in detector_ids_by_color.items():
            for det_id in ids:
                detector_color_by_global[int(det_id)] = str(color_name)
        
        # 1. Pre-split the global base model into pure X and Z base models.
        # Ensure 'split_base_dem_by_basis' is imported from your partition module.
        target_obs_basis = getattr(cfg.hybrid_two_stage, "stage2_graph_target_basis", "auto")
        ignore_hyperedges = bool(getattr(cfg, "ignore_hyperedge_error_mechanism", True))
        
        # self.base_x, self.x_global_map = split_base_dem_by_basis(
        #     base_model, self.detector_basis_by_id, "X", keep_observables=(target_obs_basis in ["X", "auto"])
        # )
        self.base_z, self.z_global_map, self.z_base_mech_map = split_base_dem_by_basis(
            base_model, self.detector_basis_by_id, "Z", keep_observables=(target_obs_basis in ["Z", "auto"])
        )

        # Lookup dictionaries to map global detector IDs to the new basis-specific local IDs
        # x_global_to_local = {g_id: l_id for l_id, g_id in enumerate(self.x_global_map)}
        z_global_to_local = {g_id: l_id for l_id, g_id in enumerate(self.z_global_map)}

        # self.x_stages = {}
        self.z_stages = {}
        # self.x_matchers = {}
        self.z_matchers = {}

        hyperedge_drops: list[tuple[str, int, int]] = []

        for c in cfg.colors:
            # 2. Filter the reduced color detectors for each basis
            color_dets_global = detector_ids_by_color[c]
            
            # color_dets_x = [x_global_to_local[d] for d in color_dets_global if d in x_global_to_local]
            color_dets_z = [z_global_to_local[d] for d in color_dets_global if d in z_global_to_local]

            # 3. Build two-stage hierarchical models for each basis independently
            # self.x_stages[c] = build_stage_models(self.base_x, color_dets_x)
            z_models = build_stage_models(self.base_z, color_dets_z)
            if ignore_hyperedges:
                pre_s1 = int(z_models.stage1.num_errors)
                pre_s2 = int(z_models.stage2.num_errors)
                z_models_dropped = drop_hyperedge_error_mechanisms_from_stage_models(z_models)
                d1 = pre_s1 - int(z_models_dropped.stage1.num_errors)
                d2 = pre_s2 - int(z_models_dropped.stage2.num_errors)
                if d1 > 0 or d2 > 0:
                    hyperedge_drops.append((str(c), int(d1), int(d2)))
                z_models = z_models_dropped
            self.z_stages[c] = z_models

            # Validate graph-likeness right before matcher construction.
            # This keeps model diagnostics close to the pymatching initialization site.
            z_stage_models = self.z_stages[c]

            def _global_detector_info(global_det_id: int) -> str:
                basis = str(self.detector_basis_by_id.get(global_det_id, "?"))
                color = str(detector_color_by_global.get(global_det_id, "?"))
                return f"global={global_det_id} basis={basis} color={color}"

            def _global_mech_detail_from_z_base_idx(z_base_idx: int) -> str:
                if z_base_idx < 0 or z_base_idx >= len(self.z_base_mech_map):
                    return f"base_mech_idx={z_base_idx} global_mech_indices=[]"

                global_mech_indices = [int(i) for i in self.z_base_mech_map[z_base_idx]]
                global_parts: list[str] = []
                for global_idx in global_mech_indices:
                    if global_idx < 0 or global_idx >= self.base_model.num_errors:
                        global_parts.append(f"global_mech_idx={global_idx} INVALID")
                        continue
                    g_prob = float(self.base_model.probabilities[global_idx])
                    g_dets = [int(d) for d in self.base_model.detector_targets[global_idx]]
                    g_info = [_global_detector_info(d) for d in g_dets]
                    global_parts.append(
                        f"global_mech_idx={global_idx} prob={g_prob:.12g} dets={g_dets} det_info={g_info}"
                    )
                return f"base_mech_idx={z_base_idx} global_mech_indices={global_mech_indices} details={global_parts}"

            def _z_stage1_detector_info(det_local_id: int) -> str:
                if det_local_id < 0 or det_local_id >= len(z_stage_models.stage1_input_detectors):
                    return f"id={det_local_id} basis=? color=?"
                z_local = int(z_stage_models.stage1_input_detectors[det_local_id])
                if z_local < 0 or z_local >= len(self.z_global_map):
                    return f"id={det_local_id} basis=? color=?"
                global_id = int(self.z_global_map[z_local])
                basis = str(self.detector_basis_by_id.get(global_id, "?"))
                color = str(detector_color_by_global.get(global_id, "?"))
                return f"id={det_local_id} global={global_id} basis={basis} color={color}"

            num_stage2_real = len(z_stage_models.stage2_real_detectors)

            def _z_stage2_detector_info(det_local_id: int) -> str:
                if det_local_id < 0:
                    return f"id={det_local_id} basis=? color=?"
                if det_local_id < num_stage2_real:
                    z_local = int(z_stage_models.stage2_real_detectors[det_local_id])
                    if z_local < 0 or z_local >= len(self.z_global_map):
                        return f"id={det_local_id} basis=? color=?"
                    global_id = int(self.z_global_map[z_local])
                    basis = str(self.detector_basis_by_id.get(global_id, "?"))
                    color = str(detector_color_by_global.get(global_id, "?"))
                    return f"id={det_local_id} global={global_id} basis={basis} color={color}"
                virtual_id = det_local_id - num_stage2_real
                return f"id={det_local_id} virtual={virtual_id} basis=VIRTUAL color=VIRTUAL"

            if not bool(getattr(cfg, "ignore_hyperedge_error_mechanism", True)):
                _raise_if_hyperedge_mechanisms(
                    stage_name=f"ConcatGraphDecoder[{c}].z.stage1.pre_matcher_init",
                    detector_targets=z_stage_models.stage1.detector_targets,
                    probabilities=z_stage_models.stage1.probabilities,
                    mech_mapping=z_stage_models.stage1_mech_mapping,
                    detector_info_resolver=_z_stage1_detector_info,
                    base_mechanism_detail_resolver=_global_mech_detail_from_z_base_idx,
                )
                _raise_if_hyperedge_mechanisms(
                    stage_name=f"ConcatGraphDecoder[{c}].z.stage2.pre_matcher_init",
                    detector_targets=z_stage_models.stage2.detector_targets,
                    probabilities=z_stage_models.stage2.probabilities,
                    mech_mapping=z_stage_models.stage2_mech_mapping,
                    detector_info_resolver=_z_stage2_detector_info,
                    base_mechanism_detail_resolver=_global_mech_detail_from_z_base_idx,
                )

            # 4. Initialize matching/UF instances
            # self.x_matchers[c] = {
            #     "stage1": _MatchingStageDecoder(self.x_stages[c].stage1, matcher_type),
            #     "stage2": _MatchingStageDecoder(self.x_stages[c].stage2, matcher_type)
            # }
            self.z_matchers[c] = {
                "stage1": _MatchingStageDecoder(self.z_stages[c].stage1, matcher_type),
                "stage2": _MatchingStageDecoder(self.z_stages[c].stage2, matcher_type)
            }

        if ignore_hyperedges:
            _warn_if_hyperedge_drops(
                cfg=cfg,
                decoder_label=self.__class__.__name__,
                basis_label="z",
                drops_by_color=hyperedge_drops,
            )

    def _decode_single_syndrome(self, c: str, syn_global: np.ndarray, s1_cache: dict, s2_cache: dict) -> tuple[np.ndarray, float, dict]:
        t0 = time.perf_counter()

        # ---------------------------------------------------------
        # X Basis Pipeline
        # ---------------------------------------------------------
        # Project global syndrome to the X-basis base space
        # syn_base_x = syn_global[self.x_global_map]
        
        # # Stage 1 X
        # syn1_x = syn_base_x[self.x_stages[c].stage1_input_detectors]
        # e1_x, _, _ = self.x_matchers[c]["stage1"].decode(syn1_x)
        
        # # Stage 2 X (e1_x aligns perfectly as virtual detectors)
        # syn2_real_x = syn_base_x[self.x_stages[c].stage2_real_detectors]
        # syn2_x = np.concatenate([syn2_real_x, e1_x.astype(np.uint8)], axis=0)
        
        # _, obs_x, cost_x = self.x_matchers[c]["stage2"].decode(syn2_x)

        # ---------------------------------------------------------
        # Z Basis Pipeline
        # ---------------------------------------------------------
        # Project global syndrome to the Z-basis base space
        syn_base_z = syn_global[self.z_global_map]
        
        # Stage 1 Z
        syn1_z = syn_base_z[self.z_stages[c].stage1_input_detectors]
        e1_z, _, cost1_z = self.z_matchers[c]["stage1"].decode(syn1_z)
        
        # Stage 2 Z (e1_z aligns perfectly as virtual detectors)
        syn2_real_z = syn_base_z[self.z_stages[c].stage2_real_detectors]
        syn2_z = np.concatenate([syn2_real_z, e1_z.astype(np.uint8)], axis=0)
        
        _, obs_z, cost_z = self.z_matchers[c]["stage2"].decode(syn2_z)

        # ---------------------------------------------------------
        # Final Combine
        # ---------------------------------------------------------
        final_obs = obs_z
        total_cost = cost_z

        stats = {
            "stage1_weight": float(cost1_z),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return final_obs, total_cost, stats

class ConcatMatchingDecoder(ConcatGraphDecoder):
    """Pure MWPM Two-Stage Decoder."""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="mwpm", **kwargs)


class ConcatFindDecoder(ConcatGraphDecoder):
    """Pure Union-Find Two-Stage Decoder."""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="uf", **kwargs)


def split_ler_by_basis(
    obs_true: np.ndarray,
    obs_pred: np.ndarray,
    basis: str,
    rounds_per_shot: int = 1,
) -> tuple[float, float]:
    mismatch = np.asarray(obs_true, dtype=np.uint8) ^ np.asarray(obs_pred, dtype=np.uint8)
    per_shot_fail = np.any(mismatch, axis=1)
    if rounds_per_shot < 1:
        raise ValueError("rounds_per_shot must be >= 1")
    ler = float(np.mean(per_shot_fail)) / float(rounds_per_shot)
    if basis == "X":
        return ler, float("nan")  
    if basis == "Z":
        return float("nan"), ler  
    raise ValueError(f"Unsupported basis: {basis}")