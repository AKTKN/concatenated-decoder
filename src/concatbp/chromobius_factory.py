from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

import stim

from .config import CircuitConfig
from .built_circuit import BuiltCircuit
from .path_setup import ensure_local_imports

# Avoid mutating sys.path unless needed (better for external packaging).
try:
    import gen  # type: ignore
    import clorco  # type: ignore
    from clorco._make_circuit import make_circuit  # type: ignore
    from clorco.color_code._color_code_layouts import (  # type: ignore
        make_color_code_layout,
        make_color_code_layout_488,
    )
    from clorco.color_code._superdense_planar_color_code_circuits import (  # type: ignore
        make_color_code_layout_for_superdense,
    )

    # If both imports succeed but come from different locations (e.g. an unrelated
    # installed `gen` package plus local `clorco`), fail over to the local monorepo
    # paths to keep APIs consistent.
    try:
        gen_base = Path(str(getattr(gen, "__file__", ""))).resolve().parents[1]
        clorco_base = Path(str(getattr(clorco, "__file__", ""))).resolve().parents[1]
        if gen_base != clorco_base:
            raise RuntimeError(f"Mixed imports detected: gen_base={gen_base} clorco_base={clorco_base}")
    except Exception:
        raise
except Exception as exc:  # pragma: no cover
    # If `gen` was imported from some other installed package before failing to
    # import `clorco`, it will be cached in sys.modules and reused even after we
    # add local paths. That leads to API mismatches such as Tile(flags=...) errors.
    for k in list(sys.modules.keys()):
        if k == "gen" or k.startswith("gen.") or k == "clorco" or k.startswith("clorco."):
            sys.modules.pop(k, None)

    ensure_local_imports()
    try:
        import gen  # type: ignore
        import clorco  # type: ignore
        from clorco._make_circuit import make_circuit  # type: ignore
        from clorco.color_code._color_code_layouts import (  # type: ignore
            make_color_code_layout,
            make_color_code_layout_488,
        )
        from clorco.color_code._superdense_planar_color_code_circuits import (  # type: ignore
            make_color_code_layout_for_superdense,
        )
    except Exception as exc2:
        raise RuntimeError(
            "Failed to import chromobius circuit-generation dependencies. "
            "Install the chromobius python package (or ensure local src paths are present)."
        ) from exc2
def _noise_model(name: str, p: float):
    if name == "si1000":
        return gen.NoiseModel.si1000(p)
    if name in {"uniform_depolarizing", "depolarizing"}:
        return gen.NoiseModel.uniform_depolarizing(p)
    if name == "none":
        return None
    if name in {"code_capacity_depolarizing", "code_capacity"}:
        return gen.NoiseModel.code_capacity_depolarizing(p)
    raise ValueError(f"Unknown noise model: {name}")


def _resolve_effective_rounds(style: str, rounds: int) -> int:
    # chromobius currently requires >=2 rounds for these styles.
    if (style.startswith("superdense_color_code_") or style.startswith("midout_color_code_")) and rounds < 2:
        return 2
    return rounds


def _is_code_capacity_noise(name: str) -> bool:
    return name in {"code_capacity_depolarizing", "code_capacity"}


def _basis_from_style(style: str) -> str | None:
    if style.endswith("_X"):
        return "X"
    if style.endswith("_Z"):
        return "Z"
    return None


def _stabilizer_code_for_style(style: str, diameter: int):
    if style.startswith("superdense_color_code_"):
        return make_color_code_layout_for_superdense(
            base_data_width=diameter,
            single_rgb_layer_instead_of_actual_code=False,
        )
    if style.startswith("midout_color_code_488_"):
        return make_color_code_layout_488(
            base_width=diameter,
            spurs="midout",
            coord_style="rect",
            single_rgb_layer_instead_of_actual_code=False,
        )
    if style.startswith("midout_color_code_"):
        return make_color_code_layout(
            base_width=diameter,
            spurs="midout",
            coord_style="rect",
            single_rgb_layer_instead_of_actual_code=False,
        )
    if style == "transit_color_code":
        return make_color_code_layout(
            base_width=diameter,
            spurs="smooth",
            coord_style="rect",
            single_rgb_layer_instead_of_actual_code=False,
        )
    if style == "transit_color_code_488":
        return make_color_code_layout_488(
            base_width=diameter,
            spurs="smooth",
            coord_style="rect",
            single_rgb_layer_instead_of_actual_code=False,
        )
    raise ValueError(
        "code-capacity mode currently supports superdense/midout/transit color-code styles. "
        f"Unsupported style: {style}"
    )


def build_chromobius_circuit(cfg: CircuitConfig, basis: Optional[str], p: float) -> BuiltCircuit:
    style = cfg.resolved_style(basis)
    if not style.startswith(("superdense_color_code_", "midout_color_code_",)):
        raise ValueError(
            "Currently, only superdense_color_code_ and midout_color_code_ styles are supported for chromobius circuits. "
            f"Unsupported style: {style}"
        )
    
    if _is_code_capacity_noise(cfg.noise_model):
        rounds_requested = cfg.resolved_noise_rounds()
        rounds_effective = rounds_requested
        code = _stabilizer_code_for_style(style, cfg.diameter)
        code_basis = basis if basis is not None else _basis_from_style(style)
        if code_basis in {"X", "Z"}:
            code = code.with_observables_from_basis(code_basis)

        if rounds_effective == 1:
            circuit = code.make_code_capacity_circuit(noise=float(p))
        else:
            circuit = code.make_phenom_circuit(
                noise=gen.NoiseRule(after={"DEPOLARIZE1": float(p)}, flip_result=0.0),
                rounds=int(rounds_effective),
            )
    else:
        rounds_requested = cfg.resolved_rounds()
        rounds_effective = _resolve_effective_rounds(style, rounds_requested)
        noise = _noise_model(cfg.noise_model, p)

        circuit = make_circuit(
            style=style,
            noise_model=noise,
            noise_strength=p,
            rounds=rounds_effective,
            diameter=cfg.diameter,
            convert_to_cz=cfg.convert_to_cz,
            editable_extras=dict(cfg.editable_extras),
            debug_out_dir=None,
        )

    return BuiltCircuit(
        style=style,
        basis="" if basis is None else basis,
        noise_strength=p,
        noise_model=cfg.noise_model,
        rounds_spec=str(cfg.rounds),
        rounds_requested=rounds_requested,
        rounds_effective=rounds_effective,
        circuit=circuit,
    )
