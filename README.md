# concatbp

Config-driven quantum memory experiments and decoder evaluation.

This repository is structured to be *git-clone + qsub friendly* on a PBS cluster.
Large experiment outputs (especially when detailed stats are enabled) are intentionally excluded from git.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Run a small smoke test
bash scripts/run_memoryexp.sh configs/threshold_postselect_smoke.json 10 10 1 quiet
```

## Running an experiment

- CLI:

```bash
PYTHONPATH=src python -m concatbp.experiment_cli --config configs/memory_simulation.json
```

- Script wrapper (recommended):

```bash
bash scripts/run_memoryexp.sh configs/memory_simulation.json
```

Outputs go under `concatbp_outputs/` (see the config `experiment.output_dir`).

## PBS (qsub)

Edit parameters in `pbs/memoryexp.pbs` (walltime, cores, config path), then:

```bash
qsub pbs/memoryexp.pbs
```

## Notes

- `scripts/*.sh` set `PYTHONPATH=src` so you can run without `pip install -e .`.
- If you use local checkouts of dependencies (e.g. modified ldpc/chromobius), the code will also try to add them via `concatbp.path_setup.ensure_local_imports()` when present.

