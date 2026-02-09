# NEXT

## Read This First
- Current focus: validate new DDP + artifact schema behavior on real HPC and synchronize docs with implemented contracts.
- Success criteria for the *next* commit: multi-GPU `torchrun` run succeeds on cluster, docs updated for mode/loss/CSV/DDP, and CSV header-order regression tests are added.

## Where to Resume
- Helper-Checkpoint: 20260209T193213Z__pipeline__ddp-loss-csv-hpc
- Helper-Note: _agent/notes/20260209T193213Z__pipeline__ddp-loss-csv-hpc.md
- Key files touched: src/run.py, src/utils/distributed.py, src/utils/data_io.py, src/utils/losses.py, src/train/base.py, src/evaluate/base.py, configs/v3.yaml, scripts/v3.sh, tests/e2e/test_hpc_pipeline_smoke.py

## Next Commit Plan (strict order)
1. Execute `scripts/v3.sh` on target HPC allocation and capture distributed runtime/artifact behavior.
2. Add/adjust tests for exact train/eval CSV header order and required columns.
3. Update `docs/design_patterns/pipeline.md`, `docs/design_patterns/evaluator.md`, and `docs/design_patterns/logging_overview.md` to match runtime behavior.
4. Decide and finalize legacy script cleanup (`scripts/run_pipeline.sh` replacement note or restore).
5. Re-run `python -m pytest` and HPC smoke gate (`RELIC_RUN_HPC_E2E=1`) where hardware is available.

## Guardrails
- What must not change: run mode contract (`full_pipeline|train_only|eval_only`), eval CSV schema without `test_` prefix, centralized loss computation path, rank-0-only checkpoint writes.
- What should be refactored later (explicitly deferred): split `src/run.py` orchestration into smaller modules once behavior stabilizes.

## Quick Retrieval
Commands the next iteration should run first:
- `git log --grep "^helper(" -n 30`
- `rg "pipeline|ddp-loss-csv-hpc|training_config.loss|EVAL_CSV_COLUMNS" /Users/richardwang/Documents/relic/_agent`
