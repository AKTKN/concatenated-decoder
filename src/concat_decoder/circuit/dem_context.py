from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Literal

import stim

from .built_circuit import BuiltCircuit
from .factory import build_circuit
from ..config import CircuitConfig
from ..dem.model import BinaryErrorModel, ParsedDetectorMetadata, parse_detector_metadata, parse_stim_dem
from ..dem.partition import (
    StageModels,
    build_stage_models,
    drop_hyperedge_error_mechanisms_from_stage_models,
    split_base_dem_by_basis,
)


Basis = Literal["X", "Z"]


@dataclass(frozen=True)
class CircuitDemContext:
    """Everything needed to derive DEMs used by decoding strategies.

    Notes:
        - `dem` and `base_model` match what `experiment.py` uses:
          `dem = circuit.detector_error_model(decompose_errors=False)` and
          `base_model = parse_stim_dem(dem)`.
        - `metadata` is parsed from detector coordinates and includes per-color
          detector ids and per-detector basis labels.
    """

    circuit_cfg: CircuitConfig
    basis: str
    p: float
    built: BuiltCircuit
    dem: stim.DetectorErrorModel
    base_model: BinaryErrorModel
    metadata: ParsedDetectorMetadata


def circuit_config_from_experiment_dict(
    experiment: Mapping[str, Any],
    *,
    distance: int,
) -> tuple[CircuitConfig, str]:
    """Builds a `CircuitConfig` from an experiment config dict.

    This accepts the same keys as the `experiment` section in concatbp JSON configs.

    Returns:
        (circuit_cfg, basis)
    """

    style = str(experiment.get("circuit_style", experiment.get("style", "superdense_color_code_{basis}")))
    circuit_from = str(experiment.get("circuit_from", "chromobius"))
    rounds = experiment.get("rounds", "d")
    noise_model = str(experiment.get("noise_model", "si1000"))
    circuit_options = dict(experiment.get("circuit_options", {}))

    # Mirror experiment.py behavior for comparative decoding: it is forwarded as
    # `enable_comparative_decoding` into circuit_options.
    if bool(experiment.get("comparative_decoding", False)):
        circuit_options = {**circuit_options, "enable_comparative_decoding": True}

    circuit_cfg = CircuitConfig(
        style=style,
        diameter=int(distance),
        rounds=rounds,
        noise_model=noise_model,  # type: ignore[arg-type]
        circuit_from=circuit_from,  # type: ignore[arg-type]
        circuit_options=circuit_options,
        convert_to_cz=bool(experiment.get("convert_to_cz", True)),
        editable_extras=dict(experiment.get("editable_extras", {})),
    )
    basis = str(experiment.get("basis", "Z"))
    return circuit_cfg, basis


def build_context_from_experiment_dict(
    experiment: Mapping[str, Any],
    *,
    distance: int,
    p: float,
    basis: str | None = None,
    decompose_errors: bool = False,
) -> CircuitDemContext:
    """Builds a circuit and its monolithic DEM (no decoding).

    This is the recommended entry point when working from JSON configs.
    """

    circuit_cfg, cfg_basis = circuit_config_from_experiment_dict(experiment, distance=int(distance))
    use_basis = cfg_basis if basis is None else str(basis)
    built = build_circuit(circuit_cfg, basis=use_basis, p=float(p))
    dem = built.circuit.detector_error_model(decompose_errors=bool(decompose_errors))
    base_model = parse_stim_dem(dem)
    metadata = parse_detector_metadata(built.circuit)

    return CircuitDemContext(
        circuit_cfg=circuit_cfg,
        basis=use_basis,
        p=float(p),
        built=built,
        dem=dem,
        base_model=base_model,
        metadata=metadata,
    )


def get_concat_stage_models_by_color(
    ctx: CircuitDemContext,
    *,
    colors: Sequence[str] = ("r", "g", "b"),
) -> dict[str, StageModels]:
    """Returns StageModels (stage1/stage2) per color from the full base DEM.

    This corresponds to the internal partitioning used by concat decoders that
    operate on the full (non basis-separated) base model.
    """

    out: dict[str, StageModels] = {}
    for c in colors:
        det_ids = ctx.metadata.detector_ids_by_color[str(c)]
        out[str(c)] = build_stage_models(ctx.base_model, det_ids)
    return out


@dataclass(frozen=True)
class BasisSeparatedStageModels:
    """StageModels built after restricting the base DEM to one measurement basis."""

    basis: Basis
    base_basis_model: BinaryErrorModel
    kept_global_detector_ids: list[int]
    stage_models_by_color: dict[str, StageModels]


def get_basis_separated_stage_models_by_color(
    ctx: CircuitDemContext,
    *,
    target_basis: Basis = "Z",
    colors: Sequence[str] = ("r", "g", "b"),
    keep_observables: bool = True,
    ignore_hyperedges: bool = True,
) -> BasisSeparatedStageModels:
    """Returns StageModels per color after splitting the base DEM by basis.

    This matches the preprocessing used by pure graph concat decoders
    (e.g. `ConcatGraphDecoder`), which work on a basis-restricted base model and
    typically drop hyperedge mechanisms for MWPM/UF compatibility.

    Args:
        target_basis: Which detector basis to keep in the basis-restricted base DEM.
        keep_observables: Whether to keep logical observables in the restricted model.
            This corresponds to `keep_observables=(target_obs_basis in [target_basis, 'auto'])`.
        ignore_hyperedges: When True, apply `drop_hyperedge_error_mechanisms_from_stage_models`
            to make the stage models graph-like.
    """

    if target_basis not in {"X", "Z"}:
        raise ValueError("target_basis must be 'X' or 'Z'")

    base_basis, kept_global_det_ids, _ = split_base_dem_by_basis(
        ctx.base_model,
        ctx.metadata.detector_basis,
        target_basis,
        keep_observables=bool(keep_observables),
    )
    global_to_local = {int(g): int(i) for i, g in enumerate(kept_global_det_ids)}

    out: dict[str, StageModels] = {}
    for c in colors:
        det_ids_global = ctx.metadata.detector_ids_by_color[str(c)]
        det_ids_local = [global_to_local[int(d)] for d in det_ids_global if int(d) in global_to_local]

        models = build_stage_models(base_basis, det_ids_local)
        if ignore_hyperedges:
            models = drop_hyperedge_error_mechanisms_from_stage_models(models)
        out[str(c)] = models

    return BasisSeparatedStageModels(
        basis=target_basis,
        base_basis_model=base_basis,
        kept_global_detector_ids=[int(x) for x in kept_global_det_ids],
        stage_models_by_color=out,
    )
