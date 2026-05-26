from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import DecoderConfig
from ..dem.model import BinaryErrorModel
from ..dem.partition import (
    build_stage_models,
    split_base_dem_by_basis,
    _belief_matching_weights,
    llr_to_prob,
    drop_hyperedge_error_mechanisms_from_stage_models,
)
from .base import (
    BaseConcatDecoder,
    DecodeBatchResult,
    _MatchingStageDecoder,
    _RelayBPStageDecoder,
    _safe_log_weights,
    _warn_if_hyperedge_drops,
)


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

        self.matcher_type = matcher_type.lower()

        self._bp_posterior_debug_prints = 0
        self._bp_failure_prior_debug_prints = 0

        self.stage1_bps = {c: _RelayBPStageDecoder(m.stage1, cfg) for c, m in self.stage_models.items()}

        target_obs_basis = getattr(cfg.hybrid_two_stage, "stage2_graph_target_basis", "Z")
        ignore_hyperedges = bool(getattr(cfg, "ignore_hyperedge_error_mechanism", True))

        self.base_z, self.z_global_map, self.z_base_mech_map = split_base_dem_by_basis(
            base_model, self.detector_basis_by_id, "Z", keep_observables=(target_obs_basis in ["Z", "auto"])
        )
        z_global_to_local = {g_id: l_id for l_id, g_id in enumerate(self.z_global_map)}

        self.fallback_z_stages = {}
        self.fallback_z_matchers = {}
        self.s1_to_z_s1_map = {}

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

            base_to_z_s1 = {}
            for z_s1_idx, z_base_dict in enumerate(z_models.stage1_mech_mapping):
                for z_base_idx in z_base_dict.keys():
                    for global_idx in self.z_base_mech_map[z_base_idx]:
                        base_to_z_s1[global_idx] = z_s1_idx

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

        stats: dict[str, Any] = {"stage1_bp_failed": 0, "decode_time_ms": 0.0, "stage1_weight": float("nan")}
        t0 = time.perf_counter()
        want_uf_stats = bool(getattr(self.cfg, "get_cluster_stats", False)) and self.matcher_type == "uf"
        uf_stage1: dict[str, Any] | None = None
        uf_stage2: dict[str, Any] | None = None

        syn1 = syn_global[models.stage1_input_detectors]
        bp_res = self.stage1_bps[c].decode_detailed(syn1)

        syn_base_z = syn_global[self.z_global_map]

        if bp_res.converged:
            e1_full = bp_res.error_prediction

            e1_z = np.zeros(z_models.stage1.num_errors, dtype=np.uint8)
            for z_s1_idx, s1_indices in enumerate(s1_to_z):
                parity = 0
                for s1_idx in s1_indices:
                    parity ^= int(e1_full[s1_idx])
                e1_z[z_s1_idx] = parity

            stats["stage1_weight"] = float(getattr(bp_res, "cost", float("nan")))
        else:
            stats["stage1_bp_failed"] = 1
            bp_llr = getattr(bp_res, "postrior_llr", getattr(bp_res, "posterior_llr", None))
            bp_probs = llr_to_prob(bp_llr)

            z_priors = np.zeros(z_models.stage1.num_errors, dtype=np.float64)
            for z_s1_idx, s1_indices in enumerate(s1_to_z):
                p_sum = sum(float(bp_probs[s1_idx]) for s1_idx in s1_indices)
                z_priors[z_s1_idx] = min(1.0, p_sum)

            if self.matcher_type == "uf":
                stage1_weights = _safe_log_weights(z_priors)
            else:
                stage1_weights = _belief_matching_weights(z_priors)

            syn1_z = syn_base_z[z_models.stage1_input_detectors]
            e1_z, _, cost1_z, uf_stage1 = z_matchers["stage1"].decode(
                syn1_z, custom_weights=stage1_weights, get_cluster_stats=want_uf_stats
            )
            stats["stage1_weight"] = float(cost1_z)

        syn2_real_z = syn_base_z[z_models.stage2_real_detectors]
        syn2_z = np.concatenate([syn2_real_z, e1_z], axis=0)

        _, obs_z, cost_z, uf_stage2 = z_matchers["stage2"].decode(
            syn2_z, get_cluster_stats=want_uf_stats
        )

        if want_uf_stats and (uf_stage1 is not None or uf_stage2 is not None):
            stats["uf_cluster_stats"] = {
                "color": str(c),
                "basis": "Z",
                "stage1": uf_stage1,
                "stage2": uf_stage2,
            }

        stats["decode_time_ms"] = (time.perf_counter() - t0) * 1000.0
        return obs_z, cost_z, stats


class ConcatBeliefMatchingDecoder(ConcatBeliefGraphDecoder):
    """Belief-Matching Hybrid: BP -> MWPM"""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="mwpm", **kwargs)
