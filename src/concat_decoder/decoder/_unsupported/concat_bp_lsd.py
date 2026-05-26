from __future__ import annotations

import time

import numpy as np

from ...config import DecoderConfig
from ...dem.model import BinaryErrorModel
from ..base import BaseConcatDecoder, _BPLSDStageDecoder


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
        e1, _, cost1 = self.stage1_decoders[c].decode(syn1)

        syn2_real = syn_global[models.stage2_real_detectors]
        syn2 = np.concatenate([syn2_real, e1.astype(np.uint8)], axis=0)

        _, obs2, cost2 = self.stage2_decoders[c].decode(syn2)

        return obs2, cost2, {
            "stage1_weight": float(cost1),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }
