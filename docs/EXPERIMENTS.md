# concatbp Experiment Runner Guide

This document explains how to run concatbp experiments from shell scripts and how to configure rounds/style/noise in a chromobius-aligned way.

## 1. Quick Start

Activate your conda environment first.

```bash
conda activate <your-env>
```

From project root:

```bash
# Memory experiment (unified runner):
PYTHONPATH=src python -m concatbp.experiment_cli

# Or via shell wrapper (recommended for repeatability):
bash scripts/run_memoryexp.sh
```

With explicit config:

```bash
PYTHONPATH=src python -m concatbp.experiment_cli --config configs/memory_simulation.json
```

Threshold with runtime overrides (without editing config):

```bash
PYTHONPATH=src python -m concatbp.experiment_cli --config configs/memory_simulation.json --shots 2000 --shots-per-batch 500
```

Threshold with multiprocessing and quiet mode:

```bash
PYTHONPATH=src python -m concatbp.experiment_cli --config configs/memory_simulation.json --shots 2000 --shots-per-batch 500 --workers 4 --quiet
```

If needed, you can override the python executable explicitly:

```bash
CONCATBP_PYTHON=/path/to/python
"$CONCATBP_PYTHON" -m concatbp.experiment_cli
```

Rounds-expression example config:

```bash
PYTHONPATH=src python -m concatbp.experiment_cli --config configs/threshold_rounds_expr_example.json
```

Arguments for threshold (`concatbp.experiment_cli`):
- `--config` (optional)
- `--shots` (optional)
- `--shots-per-batch` (optional)
- `--workers` (optional)
- `--quiet` (optional)

## 2. Config Philosophy

Circuit generation options are aligned with chromobius:
1. style: use chromobius style string (e.g. superdense_color_code_{basis})
2. noise_model: use chromobius-compatible names (si1000, uniform_depolarizing, none)

When using `experiment.circuit_from="color-code-stim"`, some circuit types use two patch sizes (`d`, `d2`).
In concatbp configs, `d` comes from `experiment.distances`, and `d2` is given via `experiment.circuit_options.d2`.

To make distance sweeps easy and conservative (without changing result formats), `circuit_options.d2` supports:
- an integer (e.g. `6`) for a fixed `d2`
- `"d"` or `"distance"` to set `d2 = d` (enables sweeps like (4,4), (6,6), ...)
- `"d+N"` (e.g. `"d+2"`) to set `d2 = d + N` (useful for `growing` / `cult+growing` where `d2 > d` is required)

Experiment options stay in concatbp configs:
1. diameter / distances
2. physical error rates p
3. shots / seed
4. output_dir
5. plot_ler_unit (`per_round`, `per_shot`, `both`)

Decoder options (`decoder` section):
1. `relay_bp`: relay-bp stage parameters (`enabled`, `gamma0`, `pre_iter`, `num_sets`, `set_max_iter`, `gamma_dist_interval`, `stop_nconv`)
2. `hybrid_two_stage.enabled`: enable hybrid relay/graphlike fallback strategy
3. `hybrid_two_stage.stage2_graph_iter_threshold`: if stage-1 relay converges at iteration <= threshold, stage-2 uses graphlike decoding
4. `hybrid_two_stage.stage2_graph_decoder`: choose `uf` or `mwpm` for graphlike stage-2 decoding
5. `hybrid_two_stage.detailed_stats`: when true, save per-shot per-color stats parquet
6. `schedule`: BP schedule for `ldpc.BpLsdDecoder` (`parallel` or `serial`; default `parallel`)
7. `color_selection_strategy`: `"all"` (default) or a list like `["r", "g"]` to restrict which color hypotheses are evaluated
8. `do_post_selection`: when true and multiple color candidates are evaluated, abort shots where candidate solutions disagree on observables; exports abort/post-selected metrics in threshold CSV

Decoder strategies:
- `chromobius`: uses the chromobius library decoder directly on the circuit's stim DEM (no two-stage splitting, no DEM modification).
  Constraint: only supported when `experiment.circuit_from` is `chromobius`.
  Note: post-selection is ignored for this strategy.

Hybrid strategy behavior:
1. Stage-1 uses relay-bp first
2. If stage-1 relay converges quickly (<= threshold), stage-2 uses graphlike UF/MWPM
3. If stage-1 relay converges but exceeds threshold, stage-2 runs relay-bp
4. If stage-1 relay does not converge, fall back to legacy BPLSD stage-1 + stage-2
5. If stage-2 relay does not converge, fall back to legacy BPLSD stage-2

Detailed stats parquet columns (`detailed_stats.parquet`):
1. `shot_index`, `color`
2. `stage1_converged_le_threshold`
3. `stage1_relay_nonconverged_bplsd`
4. `stage2_relay_nonconverged_bplsd`
5. `decode_time_ms`

Note: parquet export requires `pyarrow` in the active Python environment.

## 3. Rounds Expression

`rounds` accepts either int or distance-based expression:
1. `d` -> rounds = distance
2. `4d` -> rounds = 4 * distance
3. `d*4` -> rounds = 4 * distance
4. `4*d` -> rounds = 4 * distance
5. `7` -> rounds = 7

Examples:

```json
{
  "circuit": {
    "style": "superdense_color_code_{basis}",
    "diameter": 9,
    "rounds": "d",
    "noise_model": "si1000"
  }
}
```

```json
{
  "experiment": {
    "circuit_style": "midout_color_code_{basis}",
    "distances": [5, 7, 9],
    "rounds": "4d",
    "noise_model": "depolarizing"
  }
}
```

## 4. Supported Noise Model Strings in concatbp

1. `si1000` -> chromobius `gen.NoiseModel.si1000(p)`
2. `uniform_depolarizing` -> chromobius `gen.NoiseModel.uniform_depolarizing(p)`
3. `depolarizing` -> alias of `uniform_depolarizing`
4. `none` -> noise_model=None

## 5. Output Files

All `ler_x` / `ler_z` values in CSV are reported as **per-round** logical error rates.
Plot y-axis can be configured by `plot_ler_unit`:
1. `per_round`: per-round plot only
2. `per_shot`: per-shot plot only
3. `both`: both per-round and per-shot plots

Memory experiment output (from `concatbp.experiment_cli`):
1. `threshold_results.csv`
  - Post-selection columns (when enabled): `abort_count`, `abort_rate`, `postselected_ler_x`, `postselected_ler_z`
2. plot PNG(s):
  - `per_round` -> `threshold_concat_bplsd.png`
  - `per_shot` -> `threshold_concat_bplsd_per_shot.png`
  - `both` -> `threshold_concat_bplsd_per_round.png` and `threshold_concat_bplsd_per_shot.png`
3. `threshold_config.txt`

Threshold run is resumable when `resume=true`.

## 6. Notes

1. If style includes `{basis}`, basis is substituted from `experiment.basis`.
2. For some styles, chromobius may require a minimum rounds value; concatbp stores both requested and effective rounds in CSV.
3. Scripts use the currently active `python` (so conda environments work naturally).

## 7. Independent color-code-stim runner (`concat_mwpm`)

Use this when you want memory experiments based on the decoder implementations in `color-code-stim`, independent from concatbp decoding code.

Run:

```bash
bash scripts/run_concat_mwpm_memory.sh configs/concat_mwpm_example.json
```

Optional positional args:
1. config path (required)
2. quiet flag (optional): pass `quiet` to reduce logs

Outputs:
1. `memory_results.csv`
2. plot PNG(s):
  - `per_round` -> `ler_per_round_vs_p.png`
  - `per_shot` -> `ler_per_shot_vs_p.png`
  - `both` -> `ler_per_round_vs_p.png` and `ler_per_shot_vs_p.png`
3. `run_config.txt`

Plot axes:
1. x-axis: physical error rate `p`
2. y-axis: selected by `plot_ler_unit` (`per_round`, `per_shot`, or both)
