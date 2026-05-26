from __future__ import annotations

from typing import Any

import numpy as np
import stim

from ..base import DecodeBatchResult


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

    from pathlib import Path
    import sys

    # src-layout: <repo>/src/concat_decoder/...
    project_root = Path(__file__).resolve().parents[4]
    chromobius_dir = project_root / "chromobius"
    if chromobius_dir.exists():
        sys.modules.pop("chromobius", None)
        sys.path.insert(0, str(chromobius_dir))
        try:
            import chromobius  # type: ignore
            if hasattr(chromobius, "compile_decoder_for_dem") and hasattr(chromobius, "CompiledDecoder"):
                return chromobius
        finally:
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
        if hasattr(self._chromobius, "compile_decoder_for_dem"):
            self._decoder = self._chromobius.compile_decoder_for_dem(dem)
        else:
            self._decoder = self._chromobius.CompiledDecoder.from_dem(dem)

        self._num_detectors = int(getattr(dem, "num_detectors"))
        self._num_observables = int(getattr(dem, "num_observables"))

    def decode_batch(self, detector_outcomes: np.ndarray) -> DecodeBatchResult:
        dets = np.asarray(detector_outcomes)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

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

        return DecodeBatchResult(
            obs_prediction=obs,
            candidate_costs={"chromobius": np.zeros(obs.shape[0], dtype=np.float64)},
            detailed_stats=None,
            abort_mask=None,
        )

    def print_stage1_bp_summary(self, *, tag: str | None = None) -> None:
        return
