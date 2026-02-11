# 20260211T140754Z__pipeline__logging-csv-local-cpu-e2e

## Intent
Checkpoint structured pipeline logging + strict CSV regression coverage + local CPU E2E artifacts so next iteration can focus on real HPC validation and evidence capture.

## Change Summary
- Files changed (short): AGENTS.md, CLAUDE.md, docs/design_patterns/pipeline.md, docs/design_patterns/evaluator.md, docs/design_patterns/logging_overview.md, src/run.py, src/train/base.py, src/utils/logging.py, tests/unit/test_trainer_evaluator.py, tests/e2e/test_hpc_pipeline_smoke.py, tests/e2e/artifacts/v3_local_cpu.yaml, tests/e2e/test_local_pipeline_smoke.py, tests/integration/test_logging_artifacts.py
- Diffstat: AGENTS.md | 4 +-, CLAUDE.md | 4 +-, docs/design_patterns/evaluator.md | 17 +-, docs/design_patterns/logging_overview.md | 43 ++-, docs/design_patterns/pipeline.md | 29 +-, src/run.py | 310 +++--, src/train/base.py | 26 ++, src/utils/logging.py | 37 ++, tests/e2e/test_hpc_pipeline_smoke.py | 85 ++-, tests/unit/test_trainer_evaluator.py | 91 ++, plus new tests/e2e/artifacts/v3_local_cpu.yaml, tests/e2e/test_local_pipeline_smoke.py, tests/integration/test_logging_artifacts.py
- Public API changes: `Trainer(..., logger=None, heartbeat_every_n_steps=0)`, `Trainer.train_one_epoch(..., epoch_index=0)`, new logging helper `log_stage_event(...)`, run-config parse support for `training_config.logging.heartbeat_every_n_steps`
- Config/schema changes: added local CPU config `tests/e2e/artifacts/v3_local_cpu.yaml`; enforced strict training/eval CSV header assertions in unit/e2e tests; HPC artifact config contract now validated in `full_pipeline` mode

## Decisions & Rationale
- Decision: keep `evaluate.csv` fixed schema and `training_step.csv` header order under regression tests.
- Rationale: prevent silent artifact drift and downstream parser breakage.
- Alternatives considered: dynamic eval columns and looser header assertions.
- Risk/assumption: future metric additions require intentional contract migration.

- Decision: add optional local CPU-only E2E smoke path behind `RELIC_RUN_LOCAL_E2E=1`.
- Rationale: allows reproducible full-pipeline validation on macOS without GPU/DDP hardware.
- Alternatives considered: mock-only integration tests without real artifact writes.
- Risk/assumption: local runtime remains acceptable with tiny config.

- Decision: standardize launcher references on `scripts/v3.sh` and retire `run_pipeline.sh` references.
- Rationale: align docs and operational entrypoint with current repo state.
- Alternatives considered: keep dual launcher documentation.
- Risk/assumption: no external automation still depends on retired script name.

## State Snapshot
- Branch: scaffold
- Commit: 5f234330579ef7e1ba5ee5fce24c17fb47831c7a
- Base (optional): 2ca8255836fb6e01ce148837f67fec418b454f3c
- Uncommitted at start: yes
- Data/experiment identifiers (if any): `hpc_e2e_pretrain`, `hpc_e2e_finetune`, `hpc_e2e_eval`, `local_cpu_e2e_pretrain`, `local_cpu_e2e_finetune`, `local_cpu_e2e_eval`, seed=47

## Open Questions
- Should `RELIC_RUN_LOCAL_E2E=1` be integrated into CI as a gated nightly job?
- Do we want explicit golden artifact snapshots for `log.log` event names and ordering?
- Should `tests/e2e/artifacts/v3_local_cpu.yaml` be mirrored under `configs/` as a documented developer preset?
- Is additional rank>0 logging needed for debugging DDP hangs, while still keeping rank-0 artifact writes?

## Next Steps (actionable, ordered)
1. Run `RELIC_RUN_LOCAL_E2E=1 conda run -n relic python -m pytest tests/e2e/test_local_pipeline_smoke.py -q` and archive sample artifacts for baseline comparison.
2. Run HPC smoke with `RELIC_RUN_HPC_E2E=1` and capture `log.log` + CSV outputs from pretrain/finetune/evaluate runs.
3. Validate that stage-event logs contain required lifecycle markers on cluster (`pipeline_bootstrap`, `stage_start`, `epoch_complete`, `stage_complete`).
4. Confirm no external scripts still reference `run_pipeline.sh`; patch any remaining out-of-repo references.
5. Create next helper checkpoint after HPC evidence collection and any final contract adjustments.

## Resume Keywords
pipeline logging csv-schema EVAL_CSV_COLUMNS training_step.csv evaluate.csv heartbeat_every_n_steps local_cpu_e2e hpc_e2e v3.sh full_pipeline rank0 artifacts
