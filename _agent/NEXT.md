# NEXT

## Read This First
- Current focus: validate structured logging and strict CSV contracts on real HPC, with local CPU E2E baseline already in place.
- Success criteria for the *next* commit: HPC smoke run evidence captured, critical stage events confirmed in `log.log`, and any remaining launcher/reference mismatches resolved.

## Where to Resume
- Helper-Checkpoint: 20260211T140754Z__pipeline__logging-csv-local-cpu-e2e
- Helper-Note: _agent/notes/20260211T140754Z__pipeline__logging-csv-local-cpu-e2e.md
- Key files touched: src/run.py, src/train/base.py, src/utils/logging.py, tests/unit/test_trainer_evaluator.py, tests/e2e/test_hpc_pipeline_smoke.py, tests/e2e/test_local_pipeline_smoke.py, tests/e2e/artifacts/v3_local_cpu.yaml, docs/design_patterns/pipeline.md, docs/design_patterns/evaluator.md, docs/design_patterns/logging_overview.md

## Next Commit Plan (strict order)
1. Execute local CPU E2E gate (`RELIC_RUN_LOCAL_E2E=1`) and confirm generated artifact headers and stage log event presence.
2. Execute HPC E2E gate (`RELIC_RUN_HPC_E2E=1`) and collect `log.log`, `training_step.csv`, and `evaluate.csv` outputs for all stages.
3. Compare local vs HPC artifact contracts and patch only mismatches (no mode/schema contract expansion).
4. Sweep remaining references to retired `run_pipeline.sh` in any operational docs/scripts outside current tracked files.
5. Re-run `conda run -n relic python -m pytest -q` and checkpoint with captured evidence.

## Guardrails
- What must not change: run mode contract (`full_pipeline|train_only|eval_only`), strict eval CSV schema (`split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`), centralized `training_config.loss` path, rank-0-only artifact writes.
- What should be refactored later (explicitly deferred): split `src/run.py` orchestration into smaller modules once logging behavior stabilizes on HPC.

## Quick Retrieval
Commands the next iteration should run first:
- `git log --grep "^helper(" -n 30`
- `rg "pipeline|logging-csv-local-cpu-e2e|heartbeat_every_n_steps|EVAL_CSV_COLUMNS" /Users/richardwang/Documents/relic/_agent`
