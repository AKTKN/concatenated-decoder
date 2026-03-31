# chromobius Circuit Generation API Guide (for concatbp)

This guide documents how concatbp now follows the chromobius API for:
- syndrome-extraction/circuit style selection
- noise model selection

Distance (`diameter`), physical error rate (`p`), rounds, shots, and sweep ranges remain experiment-side parameters.

## 1. API Entry Point Used by concatbp

concatbp calls chromobius through:
- [chromobius/src/clorco/_make_circuit.py](../chromobius/src/clorco/_make_circuit.py)
- function: `make_circuit(style=..., noise_model=..., noise_strength=..., rounds=..., diameter=...)`

The `style` key determines the syndrome-extraction/circuit family.

## 2. How to Specify Syndrome Extraction Method

Use `circuit.style` (or `experiment.circuit_style`) with chromobius style names.

Examples:
- `superdense_color_code_{basis}`
- `midout_color_code_{basis}`
- `midout_color_code_488_{basis}`
- `transit_color_code`
- `phenom_color_code`
- `toric_superdense_color_code_magicEPR`

### Basis handling
If style contains `{basis}`, concatbp substitutes from `experiment.basis`.

Examples:
- style `superdense_color_code_{basis}` + basis `Z` -> `superdense_color_code_Z`

## 3. How to Specify Noise Model

Use chromobius noise model names in config:
- `si1000` -> `gen.NoiseModel.si1000(p)`
- `uniform_depolarizing` -> `gen.NoiseModel.uniform_depolarizing(p)`

Compatibility aliases accepted by concatbp:
- `depolarizing` (mapped to `uniform_depolarizing`)
- `none` (passes `noise_model=None` to chromobius)

## 4. Styles Currently Registered in chromobius

The following styles are currently discoverable from `clorco._make_circuit.CONSTRUCTIONS`:

- ablated_toric_midout_color_code_magicEPR
- ablated_toric_superdense_color_code_magicEPR
- midout_color_code_488_X
- midout_color_code_488_Z
- midout_color_code_X
- midout_color_code_Z
- mxyz_color_code
- phenom_ablated_toric_color_code
- phenom_color2surface_code
- phenom_color_code
- phenom_color_code_488
- phenom_mxyz_color_code
- phenom_pyramid_code
- phenom_rep_code
- phenom_rep_code_rbrrr
- phenom_rep_code_rg
- phenom_surface_code
- phenom_toric_color_code
- phenom_toric_pyramid_code
- phenom_toric_rep_code
- phenom_toric_rep_code_rbrrr
- phenom_toric_rep_code_rg
- phenom_toric_surface_code
- rep_code
- rep_code_rbrrr
- rep_code_rg
- superdense_color_code_X
- superdense_color_code_Z
- surface_code_X
- surface_code_Z
- surface_code_trans_cx_X
- surface_code_trans_cx_Z
- surface_code_trans_cx_magicEPR
- toric_midout_color_code_magicEPR
- toric_rep_code
- toric_rep_code_rbrrr
- toric_rep_code_rg
- toric_superdense_color_code_magicEPR
- transit_ablated_toric_color_code
- transit_color2surface_code
- transit_color_code
- transit_color_code_488
- transit_pyramid_code
- transit_rep_code
- transit_rep_code_rbrrr
- transit_rep_code_rg
- transit_surface_code
- transit_toric_color_code
- transit_toric_pyramid_code
- transit_toric_rep_code
- transit_toric_rep_code_rbrrr
- transit_toric_rep_code_rg
- transit_toric_surface_code

## 5. Notes / Constraints

- For superdense and midout families, chromobius internally requires `rounds >= 2`. concatbp keeps your requested rounds, but runs with an effective value of 2 when needed.
- For code-capacity-style circuits, chromobius exposes explicit paths via `make_code_capacity_circuit` in stabilizer-code based constructions (for example `transit_*` styles).
- In CSV outputs, concatbp now stores both requested and effective rounds, plus resolved style.

## 6. Config Examples

### Memory experiment config snippet

```json
{
  "experiment": {
    "circuit_style": "superdense_color_code_{basis}",
    "basis": "Z",
    "distances": [3, 5, 7],
    "p_min": 0.01,
    "p_max": 0.1,
    "num_p": 5,
    "noise_model": "depolarizing",
    "rounds": 1
  }
}
```
