from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import DecoderConfig
from ..dem.model import BinaryErrorModel
from ..dem.partition import (
    build_stage_models,
    split_base_dem_by_basis,
    _raise_if_hyperedge_mechanisms,
    drop_hyperedge_error_mechanisms_from_stage_models,
)
from .base import BaseConcatDecoder, DecodeBatchResult, _MatchingStageDecoder, _warn_if_hyperedge_drops


class ConcatGraphDecoder(BaseConcatDecoder):
    """
    Pure graph-based Two-Stage Decoder (No BP).
    Pre-splits the Base Model into X and Z models, processing them completely independently.
    """
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, matcher_type: str, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, **kwargs)
        self.matcher_type = str(matcher_type).strip().lower()
        if not self.detector_basis_by_id:
            raise ValueError("detector_basis_by_id must be provided for pure Graph decoders.")

        detector_color_by_global: dict[int, str] = {}
        for color_name, ids in detector_ids_by_color.items():
            for det_id in ids:
                detector_color_by_global[int(det_id)] = str(color_name)

        target_obs_basis = getattr(cfg.hybrid_two_stage, "stage2_graph_target_basis", "auto")
        ignore_hyperedges = bool(getattr(cfg, "ignore_hyperedge_error_mechanism", True))

        self.base_z, self.z_global_map, self.z_base_mech_map = split_base_dem_by_basis(
            base_model, self.detector_basis_by_id, "Z", keep_observables=(target_obs_basis in ["Z", "auto"])
        )

        z_global_to_local = {g_id: l_id for l_id, g_id in enumerate(self.z_global_map)}

        self.z_stages = {}
        self.z_matchers = {}

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
            self.z_stages[c] = z_models

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
        want_uf_stats = bool(getattr(self.cfg, "get_cluster_stats", False)) and (
            str(self.matcher_type).lower() == "uf"
        )
        uf_stage1: dict[str, Any] | None = None
        uf_stage2: dict[str, Any] | None = None

        syn_base_z = syn_global[self.z_global_map]

        syn1_z = syn_base_z[self.z_stages[c].stage1_input_detectors]
        e1_z, _, cost1_z, uf_stage1 = self.z_matchers[c]["stage1"].decode(
            syn1_z, get_cluster_stats=want_uf_stats
        )

        syn2_real_z = syn_base_z[self.z_stages[c].stage2_real_detectors]
        syn2_z = np.concatenate([syn2_real_z, e1_z.astype(np.uint8)], axis=0)

        _, obs_z, cost_z, uf_stage2 = self.z_matchers[c]["stage2"].decode(
            syn2_z, get_cluster_stats=want_uf_stats
        )

        final_obs = obs_z
        total_cost = cost_z

        stats: dict[str, Any] = {
            "stage1_weight": float(cost1_z),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }
        if want_uf_stats and (uf_stage1 is not None or uf_stage2 is not None):
            stats["uf_cluster_stats"] = {
                "color": str(c),
                "basis": "Z",
                "stage1": uf_stage1,
                "stage2": uf_stage2,
            }
        return final_obs, total_cost, stats


class ConcatMatchingDecoder(ConcatGraphDecoder):
    """Pure MWPM Two-Stage Decoder."""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="mwpm", **kwargs)
