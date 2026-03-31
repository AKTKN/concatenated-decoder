from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .dem_model import BinaryErrorModel


@dataclass(frozen=True)
class StageModels:
    """Contains the partitioned BinaryErrorModels for two-stage decoding."""
    stage1: BinaryErrorModel
    stage2: BinaryErrorModel
    stage1_input_detectors: list[int]
    stage2_real_detectors: list[int]
    stage1_mech_mapping: list[dict[int, float]]
    stage2_mech_mapping: list[dict[int, float]]

    @property
    def stage2_graph(self) -> BinaryErrorModel:
        # Keep backward compatibility: when no dedicated graph model exists,
        # expose stage2 itself.
        return self.stage2

    @property
    def stage2_graph_real_detectors(self) -> list[int]:
        return self.stage2_real_detectors

    @property
    def stage1_mechanism_to_base_indices(self) -> list[list[int]]:
        return [sorted(list(m.keys())) for m in self.stage1_mech_mapping]

    @property
    def stage2_mechanism_to_base_indices(self) -> list[list[int]]:
        return [sorted(list(m.keys())) for m in self.stage2_mech_mapping]

    @property
    def stage2_graph_mechanism_to_base_indices(self) -> list[list[int]]:
        return self.stage2_mechanism_to_base_indices


def _xor_prob(p_old: float, p_new: float) -> float:
    """Calculates the probability that exactly one of two independent events occurs.
    
    Used to combine probabilities of different physical error mechanisms that 
    result in the identical syndrome pattern (marginalization).
    """
    return p_old * (1.0 - p_new) + p_new * (1.0 - p_old)


def _raise_if_hyperedge_mechanisms(
    *,
    stage_name: str,
    detector_targets: list[list[int]],
    probabilities: np.ndarray,
    mech_mapping: list[dict[int, float]] | None = None,
    detector_info_resolver: Callable[[int], str] | None = None,
    base_mechanism_detail_resolver: Callable[[int], str] | None = None,
) -> None:
    """Raises when any mechanism flips 3+ detectors, with full diagnostics."""
    violations: list[str] = []
    for mech_idx, dets in enumerate(detector_targets):
        if len(dets) < 3:
            continue

        p = float(probabilities[mech_idx]) if mech_idx < len(probabilities) else float("nan")
        if detector_info_resolver is not None:
            det_infos = [detector_info_resolver(int(d)) for d in dets]
        else:
            det_infos = [f"id={int(d)} basis=? color=?" for d in dets]
        detail = (
            f"mech_idx={mech_idx} prob={p:.12g} det_count={len(dets)} "
            f"dets={list(map(int, dets))} det_info={det_infos}"
        )
        if mech_mapping is not None and mech_idx < len(mech_mapping):
            base_dict = mech_mapping[mech_idx]
            base_idx_sorted = sorted(int(i) for i in base_dict.keys())
            base_detail = ", ".join(
                f"{int(base_idx)}:{float(base_dict[base_idx]):.12g}" for base_idx in base_idx_sorted
            )
            detail += f" base_indices={base_idx_sorted} base_probs={{{base_detail}}}"
            if base_mechanism_detail_resolver is not None:
                global_details = [base_mechanism_detail_resolver(base_idx) for base_idx in base_idx_sorted]
                detail += f" global_base_details={global_details}"
        violations.append(detail)

    if violations:
        lines = [
            f"[{stage_name}] Found {len(violations)} non-graphlike error mechanism(s) with 3+ detector targets.",
            "Details:",
            *violations,
        ]
        raise ValueError("\n".join(lines))


def drop_hyperedge_error_mechanisms(
    model: BinaryErrorModel,
    mech_mapping: list[dict[int, float]] | None = None,
    *,
    max_detector_targets: int = 2,
) -> tuple[BinaryErrorModel, list[dict[int, float]] | None, list[int]]:
    """Drops non-graphlike mechanisms (those with > max_detector_targets detectors).

    This is intended for stages decoded via MWPM/UF which require a graph-like DEM.

    Returns:
        (new_model, new_mech_mapping, kept_mechanism_indices)
    """
    if max_detector_targets < 0:
        raise ValueError("max_detector_targets must be >= 0")

    kept: list[int] = [
        int(i) for i, dets in enumerate(model.detector_targets) if len(dets) <= max_detector_targets
    ]
    if len(kept) == model.num_errors:
        return model, mech_mapping, kept

    new_probs = np.asarray([float(model.probabilities[i]) for i in kept], dtype=np.float64)
    new_dets = [[int(d) for d in model.detector_targets[i]] for i in kept]
    new_obs = [[int(o) for o in model.observable_targets[i]] for i in kept]

    new_model = BinaryErrorModel(
        num_detectors=int(model.num_detectors),
        num_observables=int(model.num_observables),
        probabilities=new_probs,
        detector_targets=new_dets,
        observable_targets=new_obs,
    )

    if mech_mapping is None:
        return new_model, None, kept

    if len(mech_mapping) != model.num_errors:
        raise ValueError("mech_mapping length must match model.num_errors")
    new_mapping = [mech_mapping[i] for i in kept]
    return new_model, new_mapping, kept


def drop_hyperedge_error_mechanisms_from_stage_models(
    models: StageModels,
    *,
    max_detector_targets: int = 2,
) -> StageModels:
    """Drops hyperedge mechanisms from both stage1 and stage2 models.

    This helper keeps the mechanism-to-base mapping lists aligned with the filtered
    mechanism indices.
    """
    s1, s1_map, _ = drop_hyperedge_error_mechanisms(
        models.stage1,
        models.stage1_mech_mapping,
        max_detector_targets=max_detector_targets,
    )
    s2, s2_map, _ = drop_hyperedge_error_mechanisms(
        models.stage2,
        models.stage2_mech_mapping,
        max_detector_targets=max_detector_targets,
    )
    if s1_map is None or s2_map is None:
        raise ValueError("StageModels are expected to carry mech mappings")

    return StageModels(
        stage1=s1,
        stage2=s2,
        stage1_input_detectors=list(models.stage1_input_detectors),
        stage2_real_detectors=list(models.stage2_real_detectors),
        stage1_mech_mapping=list(s1_map),
        stage2_mech_mapping=list(s2_map),
    )

def split_base_dem_by_basis(
    base: BinaryErrorModel,
    detector_basis_by_id: dict[int, str],
    target_basis: str,
    keep_observables: bool = True
) -> tuple[BinaryErrorModel, list[int], list[list[int]]]:
    kept_dets = [d for d in range(base.num_detectors) if detector_basis_by_id.get(d) == target_basis]
    det_reindex = {d: i for i, d in enumerate(kept_dets)}
    
    key_to_idx: dict[tuple[frozenset[int], frozenset[int]], int] = {}
    new_dets: list[list[int]] = []
    new_obs: list[list[int]] = []
    new_probs: list[float] = []
    mech_mapping_to_base: list[list[int]] = []

    for base_idx, (p, dets, obs) in enumerate(zip(base.probabilities, base.detector_targets, base.observable_targets)):
        kept_d = [det_reindex[d] for d in dets if d in det_reindex]
        if not kept_d:
            continue
            
        kept_o = sorted(set(obs)) if keep_observables else []
        key = (frozenset(kept_d), frozenset(kept_o))
        
        if key not in key_to_idx:
            idx = len(new_dets)
            key_to_idx[key] = idx
            new_dets.append(sorted(kept_d))
            new_obs.append(kept_o)
            new_probs.append(float(p))
            mech_mapping_to_base.append([int(base_idx)])
        else:
            idx = key_to_idx[key]
            new_probs[idx] = _xor_prob(new_probs[idx], float(p))
            mech_mapping_to_base[idx].append(int(base_idx))

    basis_model = BinaryErrorModel(
        num_detectors=len(kept_dets),
        num_observables=base.num_observables,
        probabilities=np.asarray(new_probs, dtype=np.float64),
        detector_targets=new_dets,
        observable_targets=new_obs,
    )
    return basis_model, kept_dets, mech_mapping_to_base


def build_restricted_stage1_dem(
    base: BinaryErrorModel,
    ignored_color_detector_ids: list[int]
) -> tuple[BinaryErrorModel, list[dict[int, float]], dict[int, int]]:
    
    ignored_set = set(ignored_color_detector_ids)
    kept_dets = [d for d in range(base.num_detectors) if d not in ignored_set]
    det_reindex = {d: i for i, d in enumerate(kept_dets)}

    key_to_idx: dict[frozenset[int], int] = {}
    new_dets: list[list[int]] = []
    new_prob: list[float] = []

    # Mapping list from new error mechanism index to the list of original mechanism indices that were merged into it.
    mech_mapping: list[dict[int, float]] = []

    for base_idx, (p, dets, obs) in enumerate(zip(base.probabilities, base.detector_targets, base.observable_targets)):
        # Remove ignored detectors and reindex the remaining ones.
        kept_d = [det_reindex[d] for d in dets if d in det_reindex]

        if not kept_d:  # if no detectors remain after ignoring
            continue

        key = frozenset(kept_d) 

        # Check overlap and merge into one error mechanism
        if key not in key_to_idx:
            idx = len(new_dets)
            key_to_idx[key] = idx
            new_dets.append(sorted(list(key)))
            new_prob.append(float(p))

            mech_mapping.append({int(base_idx): float(p)})

        else: # if an identical syndrome pattern already exists, merge probabilities
            idx = key_to_idx[key]
            new_prob[idx] = _xor_prob(new_prob[idx], float(p))
            mech_mapping[idx][int(base_idx)] = float(p)

    stage1_model = BinaryErrorModel(
        num_detectors=len(kept_dets),
        num_observables=0,  # Logical parity tracking is deferred to Stage 2.
        probabilities=np.asarray(new_prob, dtype=np.float64),
        detector_targets=new_dets,
        observable_targets=[[] for _ in new_dets],
    )

    return stage1_model, mech_mapping, det_reindex


def build_restricted_stage2_dem(
    base: BinaryErrorModel,
    target_color_detector_ids: list[int],
    stage1_mech_mapping: list[dict[int, float]],
) -> tuple[BinaryErrorModel, list[dict[int, float]], dict[int, int]]:
    """
    Creates a stage-2 c-colored DEM by keeping only the targer color detectors (e.g. 'r') as real detectors, 
    and appending the mechanisms merged in Stage 1 as "virtual detectors".

    Args:
        base: The original BinaryErrorModel.
        target_color_detector_ids: List of detector IDs for the color kept in Stage 2.
        stage1_mech_mapping: the recovery dictionary list from stage 1.

    Returns:
        stage2_model: The Stage-2 DEM with virtual detectors included.
        stage2_mech_mapping: Recovery dictionary for Stage 2 new mechanisms 
                             -> {original base_idx: probability}.
        det_reindex: Mapping from global detector IDs to Stage-2 local detector IDs. 

    -----------------------------------------------------------------------------------------------
        Example of Data Structure Transformation:
        
        [Global Setup]
        Assume 4 global detectors: D0(Blue), D1(Green), D2(Red), D3(Red).
        Original base mechanisms:
        - e0 (idx=0, p=0.10): triggers D0, D1, D2
        - e1 (idx=1, p=0.20): triggers D0, D1, D3
        - e2 (idx=2, p=0.05): triggers D0, D1
        - e3 (idx=3, p=0.30): triggers D2
        - e4 (idx=4, p=0.01): triggers D0, D1, D2 (same as e0, different cause)

        [Stage 1 Input]
        Stage 1 ignored 'Red' (D2, D3), collapsing e0, e1, e2, and e4 into a single 
        virtual detector V0 (virt_idx=0) because they all trigger {D0, D1}.
        stage1_mech_mapping = [ {0: 0.10, 1: 0.20, 2: 0.05, 4: 0.01} ]

        [Stage 2 Processing (Keeping 'Red')]
        Stage 2 assigns local indices to real Red detectors and virtual detectors:
        - Real: D2 -> local 0, D3 -> local 1
        - Virtual: V0 -> local 2 (offset by num_real_detectors)

        The original mechanisms are projected into Stage 2 edges:
        - e0 -> triggers D2 (0) & V0 (2) => Edge [0, 2], p=0.10
        - e1 -> triggers D3 (1) & V0 (2) => Edge [1, 2], p=0.20
        - e2 -> triggers V0 (2)          => Edge [2],    p=0.05
        - e3 -> triggers D2 (0)          => Edge [0],    p=0.30
        - e4 -> triggers D2 (0) & V0 (2) => Edge [0, 2], p=0.01  <-- Merges with e0!

        [Stage 2 Output]
        Due to the merge of e0 and e4 (p = 0.10 + 0.01 = 0.11), the final output is:
        stage2_model.detector_targets = [ [0, 2], [1, 2], [2], [0] ]
        stage2_model.probabilities    = [ 0.11,   0.20,   0.05,  0.30 ]
        
        stage2_mech_mapping = [
            {0: 0.10, 4: 0.01},  # Recovery dict for Edge [0, 2]
            {1: 0.20},           # Recovery dict for Edge [1, 2]
            {2: 0.05},           # Recovery dict for Edge [2]
            {3: 0.30}            # Recovery dict for Edge [0]
        ]
    ------------------------------------------------------------------------------------------------    
    """
    # Reassign local indices for real detectors
    target_set = set(target_color_detector_ids)
    real_detectors = sorted(list(target_set))
    det_reindex = {d: i for i, d in enumerate(real_detectors)}
    num_real_detectors = len(real_detectors)
    num_virtual_detectors = len(stage1_mech_mapping)

    # Invert the stage 1 mapping dictionary
    base_to_virtual: dict[int, list[int]] = {}
    for virtual_idx, base_dict in enumerate(stage1_mech_mapping):
        for base_idx, p in base_dict.items():
            if base_idx not in base_to_virtual:
                base_to_virtual[base_idx] = []
            base_to_virtual[base_idx].append(virtual_idx)

    key_to_idx: dict[tuple[frozenset[int], frozenset[int]], int] = {}
    new_dets: list[list[int]] = []
    new_obs: list[list[int]] = []
    new_probs: list[float] = []
    stage2_mech_mapping: list[dict[int, float]] = []

    #  Iterate over all original error mechanisms to convert them for Stage 2
    for base_idx, (p, dets, obs) in enumerate(zip(base.probabilities, base.detector_targets, base.observable_targets)):
        # A. Extract real detectors 
        # Detectors other than the target color (Red) are stripped away here
        real_d = [det_reindex[d] for d in dets if d in det_reindex]
        
        # B. Extract virtual detectors
        # Get the virtual detector index that this base_idx was assigned to in Stage 1
        virt_d = [num_real_detectors + v for v in base_to_virtual.get(base_idx, [])]
        # has_virt = len(virt_d) > 0

        # if not has_virt and len(real_d) <= 2:
        #     # No virtual detectors, and ≤2 real detectors.
        #     combined_d = real_d
        # elif has_virt and len(real_d) <= 1:
        #     # Has virtual detectors, and ≤1 real detector (total will be ≤2).
        #     combined_d = sorted(set(real_d + virt_d))
        # else:
        #     # Hyperedges can't be handled by MWPM; drop.
        #     continue
        
        # C. Combine the Stage 2 detector set (Real Detectors + Virtual Detectors)
        combined_d = sorted(set(real_d + virt_d))
        
        # If the mechanism triggers neither real nor virtual detectors in Stage 2, 
        # it is completely invisible, so we skip it.
        if not combined_d:
            continue
            
        # Keep observables as they are (apply basis-specific filtering here if necessary)
        combined_obs = sorted(set(obs))
        key = (frozenset(combined_d), frozenset(combined_obs))
        
        # Detect duplicates and combine (using "addition" in this case)
        if key not in key_to_idx:
            # For new patterns
            idx = len(new_dets)
            key_to_idx[key] = idx
            new_dets.append(combined_d)
            new_obs.append(combined_obs)
            new_probs.append(float(p))
            
            # Initialize the recovery dictionary
            stage2_mech_mapping.append({base_idx: float(p)})
        else:
            # If the exact same pattern already exists (integrate redundant mechanisms)
            idx = key_to_idx[key]
            
            # As requested, probabilities are added (clipped at max physical probability 1.0)
            added_prob = new_probs[idx] + float(p)
            new_probs[idx] = min(1.0, added_prob)
            
            # Add to the recovery dictionary
            stage2_mech_mapping[idx][base_idx] = float(p)

    # Construct the Stage 2 BinaryErrorModel
    stage2_model = BinaryErrorModel(
        num_detectors=num_real_detectors + num_virtual_detectors,
        num_observables=base.num_observables,
        probabilities=np.asarray(new_probs, dtype=np.float64),
        detector_targets=new_dets,
        observable_targets=new_obs,
    )
    
    return stage2_model, stage2_mech_mapping, det_reindex

def split_stage1_dem_by_basis(
        stage1_model: BinaryErrorModel,
        stage1_input_detectors: list[int],
        detector_basis_by_id: dict[int, str],
        target_basis: str, 
) -> tuple[BinaryErrorModel, list[list[int]], list[int]]:
    """
    Splits the stage-1 dem into a basis-specific graph-like dem (X-dem or Z-dem).
    
    Args:
        stage1_model: The hypergraph dem generated for stage1.
        stage1_input_detectors: Mapping from stage 1 local detector id to global detector id.
        detector_basis_by_id: Global mapping of detector ID to its basis ('X' or 'Z').
        target_basis: The basis to keep (e.g., 'X' keeps X-basis detectors to catch Z errors).
        
    Returns:
        basis_model: The separated, graph-like DEM for the target basis.
        mech_mapping_to_stage1: Maps the new basis-edge index -> list of Stage-1 mechanism indices.
                                (Used for updating priors from BP posteriors).
        det_mapping_to_stage1: Maps the new basis-detector index -> Stage-1 local detector index.
                               (Used for slicing the syndrome).
    """

    basis_det_local_indices = []
    for s1_local_idx, global_idx in enumerate(stage1_input_detectors):
        if detector_basis_by_id.get(global_idx) == target_basis:
            basis_det_local_indices.append(s1_local_idx)
    
    det_mapping_to_stage1 = basis_det_local_indices

    s1_to_basis_reindex = {s1_idx: basis_idx for basis_idx, s1_idx in enumerate(basis_det_local_indices)}
    
    key_to_idx: dict[frozenset[int], int] = {}
    new_dets: list[list[int]] = []
    new_probs: list[float] = []
    
    # Important: record which Stage-1 mechanisms were merged into each new edge.
    mech_mapping_to_stage1: list[list[int]] = []

    # 2) Scan Stage-1 mechanisms and project into the target basis.
    for s1_idx, (p, dets, obs) in enumerate(zip(
        stage1_model.probabilities, 
        stage1_model.detector_targets, 
        stage1_model.observable_targets
    )):
        
        # Keep only detectors in the target basis (e.g., extract X-basis detectors).
        kept_d = [s1_to_basis_reindex[d] for d in dets if d in s1_to_basis_reindex]
        
        # Skip mechanisms that don't trigger any detector in this basis.
        if not kept_d:
            continue
            
        key = frozenset(kept_d)
        
        # 3) Merge duplicates (e.g., collisions between a pure-Z error and the Z component of a Y error).
        if key not in key_to_idx:
            idx = len(new_dets)
            key_to_idx[key] = idx
            new_dets.append(sorted(list(key)))
            new_probs.append(float(p))
            mech_mapping_to_stage1.append([s1_idx])
        else:
            idx = key_to_idx[key]
            # Combine probabilities (depending on how priors are produced, an additive rule could be considered).
            new_probs[idx] = _xor_prob(new_probs[idx], float(p))
            mech_mapping_to_stage1[idx].append(s1_idx)

    # 4) Build the basis DEM.
    basis_model = BinaryErrorModel(
        num_detectors=len(basis_det_local_indices),
        num_observables=stage1_model.num_observables, # Stage-1 model, so typically 0.
        probabilities=np.asarray(new_probs, dtype=np.float64),
        detector_targets=new_dets,
        observable_targets=[[] for _ in new_dets],
    )

    return basis_model, mech_mapping_to_stage1, det_mapping_to_stage1

def update_basis_dem_priors(
    bp_posteriors: np.ndarray,
    mech_mapping_to_stage1: list[list[int]]
) -> np.ndarray:
    num_basis_edges = len(mech_mapping_to_stage1)
    new_priors = np.zeros(num_basis_edges, dtype=np.float64)

    # for basis_edge_idx, s1_indices in enumerate(mech_mapping_to_stage1):
    #     p_combined = 0.0
    #     for i, s1_idx in enumerate(s1_indices):
    #         p_post = float(bp_posteriors[s1_idx])
    #         if i == 0:
    #             p_combined = p_post
    #         else:
    #             p_combined = _xor_prob(p_combined, p_post)
    #     new_priors[basis_edge_idx] = p_combined

    for basis_edge_idx, s1_indices in enumerate(mech_mapping_to_stage1):
        # Use sum of marginals (not XOR parity of independent events).
        p_sum = sum(float(bp_posteriors[s1_idx]) for s1_idx in s1_indices)
        new_priors[basis_edge_idx] = min(1.0, p_sum)

    return new_priors

def _belief_matching_weights(prob: np.ndarray) -> np.ndarray:
    """Belief-matching positive weights: w = -log(p)."""
    p = np.clip(prob, 1e-12, 1.0)
    return -np.log(p)

def llr_to_prob(llr: np.ndarray) -> np.ndarray:
    """Converts log-likelihood ratios to error probabilities."""
    llr_safe = np.asarray(llr, dtype=np.float64)
    prob = np.empty_like(llr_safe, dtype=np.float64)

    # Numerically stable sigmoid for p = 1 / (1 + exp(llr)).
    pos = llr_safe >= 0
    if np.any(pos):
        # For llr >= 0: exp(-llr) is safe.
        e = np.exp(-llr_safe[pos])
        prob[pos] = e / (1.0 + e)
    if np.any(~pos):
        # For llr < 0: exp(llr) is safe.
        e = np.exp(llr_safe[~pos])
        prob[~pos] = 1.0 / (1.0 + e)

    return prob

def split_stage2_dem_by_basis(
    stage2_model: BinaryErrorModel,
    stage2_real_detectors: list[int],
    stage1_model: BinaryErrorModel,
    stage1_input_detectors: list[int],
    detector_basis_by_id: dict[int, str],
    target_basis: str,
    keep_observables: bool,
) -> tuple[BinaryErrorModel, list[list[int]], list[int]]:
    """
    Splits the stage2-dem into a basis-specific graph-like dem (X-dem or Z-dem) for stage 2 decoding.
    Virtual detectors are dynamically assigned to the target basis if their underlying stage-1 mechanism triggered any detector of that basis.
    Args:
        stage2_model: The global Stage-2 DEM containing real and virtual detectors.
        stage2_real_detectors: Global IDs of the real detectors kept in Stage 2.
        stage1_model: The global Stage-1 DEM (used to inspect virtual detectors).
        stage1_input_detectors: Global IDs of the real detectors kept in Stage 1.
        detector_basis_by_id: Global mapping of detector ID to its basis ('X' or 'Z').
        target_basis: The basis to extract (e.g., 'X' or 'Z').
        keep_observables: Whether to keep logical observables in this split DEM.
                          (e.g., True for target_basis='Z' in a Z-memory experiment).

    Returns:
        basis_model: The separated DEM for the target basis.
        mech_mapping_to_stage2: Maps the new basis-edge index -> list of Stage-2 mechanism indices.
        det_mapping_to_stage2: Maps the new basis-detector index -> Stage-2 local detector index.
                               (Used for slicing the Stage-2 syndrome).
    """
    num_real = len(stage2_real_detectors)
    num_total = stage2_model.num_detectors

    basis_det_local_indices = []
    for s2_local_idx in range(num_total):
        if s2_local_idx < num_real:
            # for real detectors, check the global basis mapping directly
            global_id = stage2_real_detectors[s2_local_idx]
            if detector_basis_by_id.get(global_id) == target_basis:
                basis_det_local_indices.append(s2_local_idx)
        else:
            # for virtual detectors, check the basis of their consituent stage1 detectos 
            virt_idx = s2_local_idx - num_real
            s1_local_detectors = stage1_model.detector_targets[virt_idx]

            # a virtual detector is included if any of its stage1 components match the target basis
            has_target_basis = False
            for s1_local_id in s1_local_detectors:
                global_id = stage1_input_detectors[s1_local_id]
                if detector_basis_by_id.get(global_id) == target_basis:
                    has_target_basis = True
                    break
            
            if has_target_basis:
                basis_det_local_indices.append(s2_local_idx)

    # mapping to slice the syndrome during decoding
    det_mapping_to_stage2 = basis_det_local_indices
    s2_to_basis_reindex = {s2_idx: basis_idx for basis_idx, s2_idx in enumerate(basis_det_local_indices)}

    # State variables for building the new DEM
    key_to_idx: dict[tuple[frozenset[int], frozenset[int]], int] = {}
    new_dets: list[list[int]] = []
    new_obs: list[list[int]] = []
    new_probs: list[float] = []
    mech_mapping_to_stage2: list[list[int]] = []

    # 2. Project Stage-2 Mechanisms onto the Target Basis
    for s2_idx, (p, dets, obs) in enumerate(zip(
        stage2_model.probabilities, 
        stage2_model.detector_targets, 
        stage2_model.observable_targets
    )):
        
        # Filter detectors based on the target basis
        kept_d = [s2_to_basis_reindex[d] for d in dets if d in s2_to_basis_reindex]
        
        # If the mechanism triggers no detectors in this basis, it is invisible
        if not kept_d:
            continue
            
        # Handle observables based on the physical experiment constraints
        kept_obs = sorted(set(obs)) if keep_observables else []
        key = (frozenset(kept_d), frozenset(kept_obs))
        
        # 3. Merge duplicate mechanisms (Degeneracy handling)
        if key not in key_to_idx:
            idx = len(new_dets)
            key_to_idx[key] = idx
            new_dets.append(sorted(kept_d))
            new_obs.append(kept_obs)
            new_probs.append(float(p))
            mech_mapping_to_stage2.append([s2_idx])
        else:
            idx = key_to_idx[key]
            # Since Stage-2 represents disjoint error events merged from Stage 1,
            # we use additive probability (capped at 1.0 for numerical safety).
            added_prob = new_probs[idx] + float(p)
            new_probs[idx] = min(1.0, added_prob)
            mech_mapping_to_stage2[idx].append(s2_idx)

    # 4. Construct the split Basis-DEM
    basis_model = BinaryErrorModel(
        num_detectors=len(basis_det_local_indices),
        num_observables=stage2_model.num_observables,
        probabilities=np.asarray(new_probs, dtype=np.float64),
        detector_targets=new_dets,
        observable_targets=new_obs,
    )

    return basis_model, mech_mapping_to_stage2, det_mapping_to_stage2


def build_stage_models(
    base: BinaryErrorModel,
    reduced_detector_ids: list[int],
    detector_basis_by_id: dict[int, str] | None = None,
    graph_target_basis: str = "auto",
) -> StageModels:
    """
    Partitions a monolithic BinaryErrorModel into a two-stage hierarchical model.

    Stage 1 processes a subset of detectors and outputs its decisions as "virtual detectors".
    Stage 2 processes the remaining real detectors along with the virtual detectors 
    from Stage 1 to determine the final logical observables.
    """

    stage1_model, stage1_mapping, s1_det_reindex = build_restricted_stage1_dem(
        base, ignored_color_detector_ids=reduced_detector_ids
    )

    # Extract the detectors kept in stage 1
    ignored_set = set(reduced_detector_ids)
    stage1_input_detectors = [d for d in range(base.num_detectors) if d not in ignored_set]

    stage2_model, stage2_mapping, s2_det_reindex = build_restricted_stage2_dem(
        base, 
        target_color_detector_ids=reduced_detector_ids, 
        stage1_mech_mapping=stage1_mapping
    )

    stage2_real_detectors = sorted(list(set(reduced_detector_ids)))

    return StageModels(
        stage1=stage1_model,
        stage2=stage2_model,
        stage1_input_detectors=stage1_input_detectors,
        stage2_real_detectors=stage2_real_detectors,
        stage1_mech_mapping=stage1_mapping,
        stage2_mech_mapping=stage2_mapping,
    )


def _resolve_graph_target_basis(
    graph_target_basis: str,
    reduced_detector_ids: list[int],
    detector_basis_by_id: dict[int, str] | None,
) -> str:
    if graph_target_basis in {"X", "Z"}:
        return graph_target_basis

    if detector_basis_by_id is None:
        return "Z"

    n_x = 0
    n_z = 0
    for d in reduced_detector_ids:
        b = str(detector_basis_by_id.get(int(d), ""))
        if b == "X":
            n_x += 1
        elif b == "Z":
            n_z += 1

    # Prefer Z on ties so Z-memory runs are stable without extra config.
    return "X" if n_x > n_z else "Z"


def _to_graphlike_stage2_model(
    num_detectors: int,
    num_observables: int,
    probs: list[float],
    dets: list[list[int]],
    obs: list[list[int]],
    mech_to_base: list[list[int]],
) -> tuple[BinaryErrorModel, list[list[int]]]:
    graph_probs: list[float] = []
    graph_dets: list[list[int]] = []
    graph_obs: list[list[int]] = []
    graph_to_base: list[list[int]] = []

    for p, d_list, o_list, base_list in zip(probs, dets, obs, mech_to_base):
        if len(d_list) <= 2:
            graph_probs.append(float(p))
            graph_dets.append([int(d) for d in d_list])
            graph_obs.append([int(o) for o in o_list])
            graph_to_base.append([int(i) for i in base_list])
            continue

        # Star decomposition without introducing extra detectors.
        # This keeps the model graph-like for matching-based stage-2 decoding.
        scale = max(len(d_list) - 1, 1)
        p_edge = min(max(float(p) / scale, 1e-12), 1 - 1e-12)
        center = int(d_list[0])
        for n in d_list[1:]:
            graph_probs.append(p_edge)
            graph_dets.append([center, int(n)])
            graph_obs.append([int(o) for o in o_list])
            graph_to_base.append([int(i) for i in base_list])

    return (
        BinaryErrorModel(
            num_detectors=num_detectors,
            num_observables=num_observables,
            probabilities=np.asarray(graph_probs, dtype=np.float64),
            detector_targets=graph_dets,
            observable_targets=graph_obs,
        ),
        graph_to_base,
    )