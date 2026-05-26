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

from ..config import DecoderConfig
from ..dem.model import BinaryErrorModel, update_dem_priors_from_posterior_llr
from ..dem.partition import (
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
from ..utils.path_setup import ensure_local_imports


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
        "[concat-decoder] Dropped hyperedge mechanisms while building graphlike DEMs "
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


def _color_to_code(color: str) -> int:
    """Map color token to the required integer code.

    Spec: 1=red, 2=blue, 3=green.
    """
    c = str(color).strip().lower()
    if c == "r":
        return 1
    if c == "b":
        return 2
    if c == "g":
        return 3
    return 0


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

    # When UF cluster stats are enabled, decoding is performed per-unique syndrome.
    # `unique_index_by_shot` maps each shot to its unique-syndrome index.
    unique_index_by_shot: np.ndarray | None = None
    # Optional UF cluster stats for the selected (min-cost) candidate per unique syndrome.
    # Each entry is a dict carrying numpy arrays and context (e.g. color/basis/stage).
    uf_cluster_stats_by_unique: list[dict[str, Any] | None] | None = None

    # Optional UF-derived per-shot metrics (available when decoder_cfg.get_cluster_stats=True).
    # These are compact scalars intended for persistence (e.g. parquet), not full cluster arrays.
    selected_color_code: np.ndarray | None = None
    uf_stage1_max_cluster_syndrome_count: dict[str, np.ndarray] | None = None
    uf_stage2_max_cluster_syndrome_count: dict[str, np.ndarray] | None = None
    uf_stage1_max_cluster_llr: dict[str, np.ndarray] | None = None
    uf_stage2_max_cluster_llr: dict[str, np.ndarray] | None = None

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
                # ldpc's UnionFindDecoder API changed across versions:
                # - older builds: uf_method is bool with a working default
                # - newer builds (e.g. ldpc>=2.4): uf_method expects a string ('peeling'/'matrix')
                # Try the legacy constructor first (no behavior change), then fall back to explicit peeling.
                try:
                    self.decoder = UnionFindDecoder(self.h)
                except TypeError:
                    self.decoder = UnionFindDecoder(self.h, uf_method="peeling")
            else:
                self.decoder = pymatching.Matching(self.h, error_weights=self.weights)
        else:
            self.decoder = None

    def decode(
        self,
        syndrome: np.ndarray,
        custom_weights: np.ndarray | None = None,
        *,
        get_cluster_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any] | None]:
        if self.model.num_errors == 0:
            return (
                np.zeros(0, dtype=np.uint8),
                np.zeros(self.model.num_observables, dtype=np.uint8),
                0.0,
                None,
            )

        uf_stats: dict[str, Any] | None = None
        if self.decoder is None:
            e = np.zeros(self.model.num_errors, dtype=np.uint8)
            cost = 0.0
        else:
            if self.matcher_type == "uf":
                llr_weights = np.asarray(custom_weights if custom_weights is not None else self.weights, dtype=np.float64).ravel()
                if custom_weights is not None:
                    if get_cluster_stats:
                        try:
                            res = self.decoder.decode(
                                syndrome.astype(np.uint8),
                                llrs=custom_weights,
                                get_cluster_stats=True,
                            )
                        except TypeError as exc:
                            raise RuntimeError(
                                "UnionFindDecoder.decode() in this environment does not support get_cluster_stats. "
                                "Install/use the patched ldpc from this repo (ldpc_v2_modifiedUF) or a compatible ldpc build."
                            ) from exc
                        if isinstance(res, np.ndarray):
                            # ldpc_v2_modifiedUF returns only the decoding for the zero-syndrome fast-path.
                            e = np.asarray(res, dtype=np.uint8).ravel()
                            uf_stats = {
                                "cluster_syndrome_counts": np.zeros(0, dtype=np.int32),
                                "cluster_error_indptr": np.zeros(1, dtype=np.int32),
                                "cluster_error_indices": np.zeros(0, dtype=np.int32),
                                "decode_time_us": 0,
                            }
                        else:
                            # (decoding, cluster_syndrome_counts, cluster_error_indptr, cluster_error_indices, decode_time_us)
                            e = np.asarray(res[0], dtype=np.uint8).ravel()
                            uf_stats = {
                                "cluster_syndrome_counts": np.asarray(res[1], dtype=np.int32),
                                "cluster_error_indptr": np.asarray(res[2], dtype=np.int32),
                                "cluster_error_indices": np.asarray(res[3], dtype=np.int32),
                                "decode_time_us": int(res[4]),
                            }
                    else:
                        e = np.asarray(
                            self.decoder.decode(syndrome.astype(np.uint8), llrs=custom_weights),
                            dtype=np.uint8,
                        ).ravel()
                    cost = float(np.dot(custom_weights, e))
                else:
                    # Use model-derived priors by default.
                    if get_cluster_stats:
                        try:
                            res = self.decoder.decode(
                                syndrome.astype(np.uint8),
                                llrs=self.weights,
                                get_cluster_stats=True,
                            )
                        except TypeError as exc:
                            raise RuntimeError(
                                "UnionFindDecoder.decode() in this environment does not support get_cluster_stats. "
                                "Install/use the patched ldpc from this repo (ldpc_v2_modifiedUF) or a compatible ldpc build."
                            ) from exc
                        if isinstance(res, np.ndarray):
                            e = np.asarray(res, dtype=np.uint8).ravel()
                            uf_stats = {
                                "cluster_syndrome_counts": np.zeros(0, dtype=np.int32),
                                "cluster_error_indptr": np.zeros(1, dtype=np.int32),
                                "cluster_error_indices": np.zeros(0, dtype=np.int32),
                                "decode_time_us": 0,
                            }
                        else:
                            e = np.asarray(res[0], dtype=np.uint8).ravel()
                            uf_stats = {
                                "cluster_syndrome_counts": np.asarray(res[1], dtype=np.int32),
                                "cluster_error_indptr": np.asarray(res[2], dtype=np.int32),
                                "cluster_error_indices": np.asarray(res[3], dtype=np.int32),
                                "decode_time_us": int(res[4]),
                            }
                    else:
                        e = np.asarray(
                            self.decoder.decode(syndrome.astype(np.uint8), llrs=self.weights),
                            dtype=np.uint8,
                        ).ravel()
                    cost = float(np.dot(self.weights, e))

                if get_cluster_stats and uf_stats is not None:
                    counts = np.asarray(
                        uf_stats.get("cluster_syndrome_counts", np.zeros(0, dtype=np.int32)),
                        dtype=np.int32,
                    ).ravel()
                    indptr = np.asarray(
                        uf_stats.get("cluster_error_indptr", np.zeros(1, dtype=np.int32)),
                        dtype=np.int32,
                    ).ravel()
                    indices = np.asarray(
                        uf_stats.get("cluster_error_indices", np.zeros(0, dtype=np.int32)),
                        dtype=np.int32,
                    ).ravel()

                    max_syn = int(np.max(counts)) if counts.size else 0

                    max_llr = 0.0
                    if indptr.size >= 2 and indices.size:
                        for k in range(int(indptr.size) - 1):
                            a = int(indptr[k])
                            b = int(indptr[k + 1])
                            if a >= b:
                                s = 0.0
                            else:
                                s = float(np.sum(llr_weights[indices[a:b]]))
                            if s > max_llr:
                                max_llr = s

                    uf_stats["max_cluster_syndrome_count"] = max_syn
                    uf_stats["max_cluster_llr"] = float(max_llr)
            else:
                if custom_weights is not None:
                    # Dynamically re-instantiate matching graph for updated BP priors
                    temp_decoder = pymatching.Matching(self.h, error_weights=custom_weights)
                    e = np.asarray(temp_decoder.decode(syndrome.astype(bool)), dtype=np.uint8).ravel()
                    cost = float(np.dot(custom_weights, e))
                else:
                    e = np.asarray(self.decoder.decode(syndrome.astype(bool)), dtype=np.uint8).ravel()
                    cost = float(np.dot(self.weights, e))

        obs = np.asarray((self.o @ e) % 2, dtype=np.uint8).ravel()
        return e, obs, cost, uf_stats

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

        want_uf_stats = bool(getattr(self.cfg, "get_cluster_stats", False))
        best_uf_stats_unique: list[dict[str, Any] | None] | None = [None] * int(num_unique) if want_uf_stats else None

        selected_color_code_unique: np.ndarray | None = None
        uf_s1_max_syn_unique: dict[str, np.ndarray] | None = None
        uf_s2_max_syn_unique: dict[str, np.ndarray] | None = None
        uf_s1_max_llr_unique: dict[str, np.ndarray] | None = None
        uf_s2_max_llr_unique: dict[str, np.ndarray] | None = None
        if want_uf_stats:
            selected_color_code_unique = np.full(num_unique, 0, dtype=np.int8)
            uf_s1_max_syn_unique = {c: np.full(num_unique, -1, dtype=np.int32) for c in colors}
            uf_s2_max_syn_unique = {c: np.full(num_unique, -1, dtype=np.int32) for c in colors}
            uf_s1_max_llr_unique = {c: np.full(num_unique, np.nan, dtype=np.float64) for c in colors}
            uf_s2_max_llr_unique = {c: np.full(num_unique, np.nan, dtype=np.float64) for c in colors}

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

                if want_uf_stats and isinstance(stats, dict) and uf_s1_max_syn_unique is not None:
                    uf = stats.get("uf_cluster_stats")
                    if isinstance(uf, dict):
                        s1 = uf.get("stage1")
                        s2 = uf.get("stage2")
                        if isinstance(s1, dict):
                            uf_s1_max_syn_unique[str(c)][i] = int(s1.get("max_cluster_syndrome_count", 0))
                            assert uf_s1_max_llr_unique is not None
                            uf_s1_max_llr_unique[str(c)][i] = float(s1.get("max_cluster_llr", 0.0))
                        if isinstance(s2, dict):
                            assert uf_s2_max_syn_unique is not None
                            uf_s2_max_syn_unique[str(c)][i] = int(s2.get("max_cluster_syndrome_count", 0))
                            assert uf_s2_max_llr_unique is not None
                            uf_s2_max_llr_unique[str(c)][i] = float(s2.get("max_cluster_llr", 0.0))
                if total_cost < best_cost_unique[i]:
                    best_cost_unique[i] = total_cost
                    best_obs_unique[i] = obs2
                    best_color_unique[i] = int(color_to_index[str(c)])
                    if selected_color_code_unique is not None:
                        selected_color_code_unique[i] = np.int8(_color_to_code(str(c)))
                    if iterations_unique is not None and shot_iteration is not None:
                        iterations_unique[i] = shot_iteration
                    if want_uf_stats and best_uf_stats_unique is not None:
                        best_uf_stats_unique[i] = stats.get("uf_cluster_stats") if isinstance(stats, dict) else None

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
            unique_index_by_shot=inverse if want_uf_stats else None,
            uf_cluster_stats_by_unique=best_uf_stats_unique,
            selected_color_code=(selected_color_code_unique[inverse] if selected_color_code_unique is not None else None),
            uf_stage1_max_cluster_syndrome_count=(
                {c: uf_s1_max_syn_unique[c][inverse] for c in colors} if uf_s1_max_syn_unique is not None else None
            ),
            uf_stage2_max_cluster_syndrome_count=(
                {c: uf_s2_max_syn_unique[c][inverse] for c in colors} if uf_s2_max_syn_unique is not None else None
            ),
            uf_stage1_max_cluster_llr=(
                {c: uf_s1_max_llr_unique[c][inverse] for c in colors} if uf_s1_max_llr_unique is not None else None
            ),
            uf_stage2_max_cluster_llr=(
                {c: uf_s2_max_llr_unique[c][inverse] for c in colors} if uf_s2_max_llr_unique is not None else None
            ),
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
        print(f"[concat-decoder][stage1_bp_nonconvergence_summary] {{" + ", ".join(parts) + "}}")
