from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import stim


def llr_to_prob(llr: np.ndarray, *, clip_eps: float = 1e-12) -> np.ndarray:
    """Converts LLR values to probabilities with stable clipping."""
    llr_arr = np.asarray(llr, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(llr_arr))
    return np.clip(probs, clip_eps, 1.0 - clip_eps)


def update_dem_priors_from_posterior_llr(
    model: "BinaryErrorModel",
    posterior_llr: np.ndarray,
    *,
    clip_eps: float = 1e-12,
) -> "BinaryErrorModel":
    """Returns a copy of the DEM with mechanism priors replaced from posterior LLRs."""
    posterior_probs = llr_to_prob(np.asarray(posterior_llr, dtype=np.float64), clip_eps=clip_eps)
    if posterior_probs.ndim != 1:
        raise ValueError("posterior_llr must be a 1D array")
    if posterior_probs.shape[0] != model.num_errors:
        raise ValueError(
            f"posterior_llr length mismatch: got {posterior_probs.shape[0]}, expected {model.num_errors}"
        )

    return BinaryErrorModel(
        num_detectors=int(model.num_detectors),
        num_observables=int(model.num_observables),
        probabilities=posterior_probs.astype(np.float64, copy=False),
        detector_targets=[[int(d) for d in dets] for dets in model.detector_targets],
        observable_targets=[[int(o) for o in obs] for obs in model.observable_targets],
    )


def _xor_sets(sets: Iterable[list[int]]) -> list[int]:
    out: set[int] = set()
    for s in sets:
        ss = set(s)
        out = (out - ss) | (ss - out)
    return sorted(out)


@dataclass(frozen=True)
class BinaryErrorModel:
    num_detectors: int
    num_observables: int
    probabilities: np.ndarray
    detector_targets: list[list[int]]
    observable_targets: list[list[int]]

    def __post_init__(self) -> None:
        if len(self.detector_targets) != len(self.observable_targets):
            raise ValueError("detector_targets and observable_targets length mismatch")
        if len(self.probabilities) != len(self.detector_targets):
            raise ValueError("probabilities and mechanism count mismatch")

    @property
    def num_errors(self) -> int:
        return len(self.detector_targets)

    def to_matrices(self) -> tuple[sp.csc_matrix, sp.csc_matrix]:
        det_rows: list[int] = []
        det_cols: list[int] = []
        obs_rows: list[int] = []
        obs_cols: list[int] = []
        for col, dets in enumerate(self.detector_targets):
            for d in dets:
                det_rows.append(int(d))
                det_cols.append(col)
        for col, obs in enumerate(self.observable_targets):
            for o in obs:
                obs_rows.append(int(o))
                obs_cols.append(col)

        data_det = np.ones(len(det_rows), dtype=np.uint8)
        data_obs = np.ones(len(obs_rows), dtype=np.uint8)
        h = sp.csc_matrix(
            (data_det, (det_rows, det_cols)),
            shape=(self.num_detectors, self.num_errors),
            dtype=np.uint8,
        )
        o = sp.csc_matrix(
            (data_obs, (obs_rows, obs_cols)),
            shape=(self.num_observables, self.num_errors),
            dtype=np.uint8,
        )
        return h, o


@dataclass(frozen=True)
class ParsedDetectorMetadata:
    detector_color: dict[int, str]
    detector_basis: dict[int, str]
    detector_ids_by_color: dict[str, list[int]]

def parse_detector_metadata(circuit: stim.Circuit) -> ParsedDetectorMetadata:
    """
    Parses the detector metadata from a stim circuit.
    supports both Chromobius format and color-code-stim format.
    Chromobius format:
      Coordinate length is typically 4: (x, y, t, combined_val)
      combined_val: 0~2 for X basis (r,g,b), 3~5 for Z basis (r,g,b)
      
    color-code-stim format:
      Coordinate length is 5 or more: (x, y, t, pauli_val, color_val)
      pauli_val: 0=X, 1=Y, 2=Z
      color_val: 0=r, 1=g, 2=b
    """
    coords = circuit.get_detector_coordinates()
    detector_color: dict[int, str] = {}
    detector_basis: dict[int, str] = {}
    detector_ids_by_color: dict[str, list[int]] = {"r": [], "g": [], "b": []}

    # Mapping for Chromobius detector coordinate encoding.
    chromobius_color_map = {0: "r", 1: "g", 2: "b", 3: "r", 4: "g", 5: "b"}
    chromobius_basis_map = {0: "X", 1: "X", 2: "X", 3: "Z", 4: "Z", 5: "Z"}

    # Mapping for the newer color-code-stim style detector coordinates.
    new_color_map = {0: "r", 1: "g", 2: "b"}
    new_basis_map = {0: "X", 1: "Y", 2: "Z"}

    for det_id, c in coords.items():
        if len(c) >= 5:
            # color-code-stim
            pauli_val = int(round(c[3]))
            color_val = int(round(c[4]))
            
            if pauli_val not in new_basis_map or color_val not in new_color_map:
                continue
                
            col = new_color_map[color_val]
            bas = new_basis_map[pauli_val]
            
        elif len(c) == 4:
            # Chromobius
            k = int(round(c[3]))
            
            if k not in chromobius_color_map:
                continue
                
            col = chromobius_color_map[k]
            bas = chromobius_basis_map[k]
            
        else:
            raise ValueError(f"Unexpected detector coordinate length={len(c)}")

        detector_color[det_id] = col
        detector_basis[det_id] = bas
        detector_ids_by_color[col].append(det_id)

    for k in detector_ids_by_color:
        detector_ids_by_color[k].sort()
        
    return ParsedDetectorMetadata(
        detector_color=detector_color,
        detector_basis=detector_basis,
        detector_ids_by_color=detector_ids_by_color,
    )

def parse_chromobius_detector_metadata(circuit: stim.Circuit) -> ParsedDetectorMetadata:
    """
    Parses the detector metadata from a stim.Circuit's get_detector_coordinates() output.
    The metadata is expected to be encoded in the 4th coordinate (index 3) of the detector coordinates (see chromobius getting started).
    ParsedDetectorMetadata(
        # detector_color: maps detector id -> color label
        detector_color={
            0: 'r',
            1: 'g',
            2: 'b',
            3: 'r',
            ...
        },
        # detector_basis: maps detector id -> measured parity basis
        detector_basis={
            0: 'X',
            1: 'X',
            2: 'X',
            3: 'Z',
            ...
        },
        # detector_ids_by_color: color label -> sorted detector ids
        detector_ids_by_color={
            'r': [0, 3, 6, 9, ...],
            'g': [1, 4, 7, 10, ...],
            'b': [2, 5, 8, 11, ...]
        }
    )
    """
    coords = circuit.get_detector_coordinates()
    detector_color: dict[int, str] = {}
    detector_basis: dict[int, str] = {}
    detector_ids_by_color: dict[str, list[int]] = {"r": [], "g": [], "b": []}

    color_map = {0: "r", 1: "g", 2: "b", 3: "r", 4: "g", 5: "b"}
    basis_map = {0: "X", 1: "X", 2: "X", 3: "Z", 4: "Z", 5: "Z"}

    for det_id, c in coords.items():
        if len(c) < 4:
            continue
        k = int(round(c[3]))
        if k not in color_map:
            continue
        col = color_map[k]
        bas = basis_map[k]
        detector_color[det_id] = col
        detector_basis[det_id] = bas
        detector_ids_by_color[col].append(det_id)

    for k in detector_ids_by_color:
        detector_ids_by_color[k].sort()
    return ParsedDetectorMetadata(
        detector_color=detector_color,
        detector_basis=detector_basis,
        detector_ids_by_color=detector_ids_by_color,
    )




def parse_stim_dem(dem: stim.DetectorErrorModel) -> BinaryErrorModel:
    """
    Parses a stim.DetectorErrorModel into a BinaryErrorModel.
    Return a BinaryErrorModel where each mechanism corresponds to a single line in the original DEM.
    e.g. 
        BinaryErrorModel(
            num_detectors=100,  
            num_observables=1,  
            probabilities=np.array([0.01, 0.01, ...]), # 1D numpy array
            detector_targets=[[0, 1], [2, 3], [0, 1, 2, 3], ...], 
            observable_targets=[[], [], [0], ...]
        )
    """
    probs: list[float] = []
    det_targets: list[list[int]] = []
    obs_targets: list[list[int]] = []

    for inst in dem.flattened():
        if inst.type != "error":
            continue

        p = float(inst.args_copy()[0])
        det_groups: list[list[int]] = [[]]
        obs_groups: list[list[int]] = [[]]

        for t in inst.targets_copy():
            if t.is_separator():
                det_groups.append([])
                obs_groups.append([])
            elif t.is_relative_detector_id():
                det_groups[-1].append(int(t.val))
            elif t.is_logical_observable_id():
                obs_groups[-1].append(int(t.val))

        # Keep each decomposed component as its own mechanism.
        # This matches color-code-stim's handling and avoids collapsing
        # separator-decomposed graph-like parts back into hyperedges.
        for det_group, obs_group in zip(det_groups, obs_groups):
            dets = _xor_sets([det_group])
            obs = _xor_sets([obs_group])
            if not dets and not obs:
                continue
            probs.append(p)
            det_targets.append(dets)
            obs_targets.append(obs)

    return BinaryErrorModel(
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        probabilities=np.asarray(probs, dtype=np.float64),
        detector_targets=det_targets,
        observable_targets=obs_targets,
    )
