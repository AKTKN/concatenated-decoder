from __future__ import annotations

import numpy as np

from ...config import DecoderConfig
from ...dem.model import BinaryErrorModel, update_dem_priors_from_posterior_llr
from ..base import DecodeBatchResult, _RelayBPStageDecoder
from ..concat_mwpm import ConcatMatchingDecoder


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
            "[concat-decoder][belief_concatmwpm_bp_summary] "
            f"{prefix}failed={failed}/{attempted} ({rate:.6g})"
        )
