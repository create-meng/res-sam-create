# Experiments Directory

## Active Entry Points

- `_feature_bank.py`
- `_inference_auto.py`
- `_evaluate.py`
- `run_ablation_main.py`

## Shared Modules

- `dataset_layout.py`
- `paper_constants.py`
- `resize_policy.py`

## Legacy

- `legacy/v29/`
  Historical V29 pipeline snapshots.
- `legacy/sweeps/`
  Older quick-sweep and parallel-sweep runners kept only for traceability.

Rule:
- Root `experiments/` only keeps the current mainline and the paper-facing ablation entry.
- Older version scripts go under `legacy/` unless they are promoted back to the active pipeline.

## Run Modes

- Default:
  `python experiments/run_ablation_main.py`
- Two-worker supervised parallel run:
  `RUN_PARALLEL=1 PARALLEL_WORKERS=2 python experiments/run_ablation_main.py`

The parallel mode prebuilds the shared current feature bank once, splits cases across workers, and restarts a failed worker so the underlying checkpoint files can continue from the last saved position.
