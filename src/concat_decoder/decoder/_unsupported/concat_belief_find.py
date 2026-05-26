from __future__ import annotations

from ...config import DecoderConfig
from ...dem.model import BinaryErrorModel
from ..concat_bp_matching import ConcatBeliefGraphDecoder


class ConcatBeliefFindDecoder(ConcatBeliefGraphDecoder):
    """Belief-Find Hybrid: BP -> Union-Find"""
    def __init__(self, base_model: BinaryErrorModel, detector_ids_by_color: dict[str, list[int]], cfg: DecoderConfig, **kwargs):
        super().__init__(base_model, detector_ids_by_color, cfg, matcher_type="uf", **kwargs)
