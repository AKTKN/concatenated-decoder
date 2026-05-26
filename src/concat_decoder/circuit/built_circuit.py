from __future__ import annotations

from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class BuiltCircuit:
    style: str
    basis: str
    noise_strength: float
    noise_model: str
    rounds_spec: str
    rounds_requested: int
    rounds_effective: int
    circuit: stim.Circuit
