from __future__ import annotations

import time

import numpy as np

from ...config import DecoderConfig
from ...dem.model import BinaryErrorModel
from ..base import BaseConcatDecoder, _RelayBPStageDecoder, relay_bp


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

        res1 = self.stage1_decoders[c].decode_detailed(syn1)
        e1_virtual = res1.error_prediction

        syn2_real = syn_global[models.stage2_real_detectors]
        syn2 = np.concatenate([syn2_real, e1_virtual.astype(np.uint8)], axis=0)

        res2 = self.stage2_decoders[c].decode_detailed(syn2)

        stats = {
            "stage1_converged": int(res1.converged),
            "stage2_converged": int(res2.converged),
            "stage1_weight": float(res1.cost),
            "iteration": int(res1.iterations) + int(res2.iterations),
            "decode_time_ms": (time.perf_counter() - t0) * 1000.0,
        }

        return res2.observables_prediction, res2.cost, stats
