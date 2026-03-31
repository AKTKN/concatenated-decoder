from __future__ import annotations

import importlib
import re
from typing import Any, Optional

from .built_circuit import BuiltCircuit
from .config import CircuitConfig
from .path_setup import ensure_local_imports


def _build_color_code_stim_noise_model(noise_name: str, p: float):
	"""Map concatbp noise names to color-code-stim NoiseModel."""
	mod = importlib.import_module("color_code_stim")
	NoiseModel = mod.NoiseModel

	if noise_name in {"depolarizing", "uniform_depolarizing"}:
		return NoiseModel.uniform_circuit_noise(float(p))

	if noise_name in {"code_capacity_depolarizing", "code_capacity"}:
		return NoiseModel(depol=float(p))

	if noise_name == "none":
		return NoiseModel()

	raise ValueError(
		"Unsupported noise_model for color-code-stim backend: "
		f"{noise_name}. Supported values: depolarizing, uniform_depolarizing, "
		"code_capacity_depolarizing, code_capacity, none."
	)


def _infer_color_code_stim_circuit_type(style: str) -> str:
	# Current concatbp memory experiments are triangular styles.
	if style.startswith("superdense_color_code_"):
		return "tri"
	if style.startswith("midout_color_code_"):
		return "tri"
	if style.startswith("transit_color_code"):
		return "tri"
	if style.startswith("trioptimal_color_code_"):
		return "tri"
	return "tri"


def _resolve_color_code_stim_d2(raw: Any, *, d: int) -> int:
	"""Resolve color-code-stim's d2 parameter from concatbp circuit_options.

	Supported forms:
	- int: used as-is
	- str: "d" / "distance" (same as d), or "d+N" (integer offset)
	- None: caller should treat as "unset" (i.e. omit from kwargs)
	"""
	if raw is None:
		raise ValueError("d2 is null; omit circuit_options.d2 instead of using null")
	if isinstance(raw, bool):
		# Prevent bool being treated as int.
		raise ValueError(f"d2 must be an int or str, got bool: {raw!r}")
	if isinstance(raw, int):
		return int(raw)
	if isinstance(raw, str):
		tok = raw.strip().lower()
		if tok in {"d", "distance"}:
			return int(d)
		m = re.fullmatch(r"d\s*\+\s*(\d+)", tok)
		if m:
			return int(d) + int(m.group(1))
		raise ValueError(
			"Unsupported d2 string format. Use an integer, 'd'/'distance', or 'd+N'. "
			f"Got d2={raw!r}"
		)
	raise ValueError(
		"d2 must be an int or str (e.g. 6, 'd', 'd+2'). "
		f"Got type={type(raw).__name__} value={raw!r}"
	)

	
def _build_color_code_stim_circuit(
	cfg: CircuitConfig,
	basis: Optional[str],
	p: float,
) -> BuiltCircuit:
	"""Build a stim circuit through color-code-stim's ColorCode class.

	Design notes:
	- Keeps concatbp-facing options local (`circuit_options`) and maps them here.
	- Uses conservative defaults so existing threshold configs can be extended later.
	"""
	ensure_local_imports()

	mod = importlib.import_module("color_code_stim")
	ColorCode = mod.ColorCode

	style = cfg.resolved_style(basis)
	rounds_requested = cfg.resolved_rounds()
	rounds_effective = rounds_requested

	noise_model = _build_color_code_stim_noise_model(cfg.noise_model, float(p))

	options = dict(cfg.circuit_options)
	extra_kwargs = dict(options.get("extra_color_code_kwargs", {}))

	# concatbp-local option names
	layout = str(options.get("layout", _infer_color_code_stim_circuit_type(style)))
	cnot_schedule = options.get("cnot_schedule", "tri_optimal")
	superdense = bool(
		options.get("superdense", style.startswith("superdense_color_code_"))
	)

	# color-code-stim special cases:
	# - For 'cult+growing' and 'rec_stability', temp_bdry_type must be omitted
	#   (i.e. None) so ColorCode can choose its own defaults.
	#   Passing it triggers an AssertionError inside ColorCode.__init__.
	layout_norm = layout.replace(" ", "").lower()
	if layout_norm in {"cult+growing", "cultivation+growing", "rec_stability"}:
		if "temporal_boundary" in options and options.get("temporal_boundary") is not None:
			raise ValueError(
				"circuit_options.temporal_boundary must be omitted (or null) when layout is "
				"'cult+growing' or 'rec_stability' (color-code-stim chooses the boundary automatically)."
			)
		temporal_boundary = None
	else:
		temporal_boundary = options.get("temporal_boundary", basis)

	color_code_kwargs: dict[str, Any] = {
		"d": int(cfg.diameter),
		"rounds": int(rounds_requested),
		"circuit_type": layout,
		"noise_model": noise_model,
		"superdense_circuit": superdense,
		"cnot_schedule": cnot_schedule,
		"perfect_logical_initialization": bool(
			options.get("perfect_state_prep", False)
		),
		"perfect_logical_measurement": bool(
			options.get("perfect_final_measurement", False)
		),
		"perfect_first_syndrome_extraction": bool(
			options.get("perfect_first_round", False)
		),
		"comparative_decoding": bool(
			options.get("enable_comparative_decoding", False)
		),
	}

	if temporal_boundary is not None:
		color_code_kwargs["temp_bdry_type"] = str(temporal_boundary)

	if "d2" in options:
		raw_d2 = options.get("d2")
		if raw_d2 is not None:
			color_code_kwargs["d2"] = _resolve_color_code_stim_d2(
				raw_d2, d=int(cfg.diameter)
			)
	if "cnot_schedule" in options:
		color_code_kwargs["cnot_schedule"] = options["cnot_schedule"]

	color_code_kwargs.update(extra_kwargs)

	color_code = ColorCode(**color_code_kwargs)

	return BuiltCircuit(
		style=style,
		basis="" if basis is None else basis,
		noise_strength=float(p),
		noise_model=cfg.noise_model,
		rounds_spec=str(cfg.rounds),
		rounds_requested=int(rounds_requested),
		rounds_effective=int(rounds_effective),
		circuit=color_code.circuit,
	)


def build_circuit(cfg: CircuitConfig, basis: Optional[str], p: float) -> BuiltCircuit:
	"""Build a circuit from the selected backend.

	- `chromobius`: existing behavior.
	- `color-code-stim`: use ColorCode-based generation.
	"""
	source = str(cfg.circuit_from)
	if source == "chromobius":
		# Lazy import to avoid importing chromobius generation deps when unused.
		from .chromobius_factory import build_chromobius_circuit
		return build_chromobius_circuit(cfg, basis=basis, p=p)
	if source == "color-code-stim":
		return _build_color_code_stim_circuit(cfg, basis=basis, p=p)
	raise ValueError(
		f"Unknown circuit_from='{source}'. Use 'chromobius' or 'color-code-stim'."
	)
