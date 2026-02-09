# 20260209T193213Z__pipeline__ddp-loss-csv-hpc

## Intent
Checkpoint pipeline alignment changes for documented modes, CSV schemas, configurable loss, DDP runtime, and HPC/e2e artifacts so the next iteration can focus on cluster validation and doc sync.

## Change Summary
- Files changed (short): configs/v3.yaml, scripts/run_pipeline.sh, scripts/v3.sh, src/evaluate/base.py, src/run.py, src/train/base.py, src/train/strategies.py, src/utils/data_io.py, src/utils/distributed.py, src/utils/logging.py, src/utils/losses.py, tests/e2e/artifacts/v3_hpc.yaml, tests/e2e/test_hpc_pipeline_smoke.py, tests/integration/test_run_pipeline_modes.py, tests/unit/test_losses.py, tests/unit/test_trainer_evaluator.py
- Diffstat: configs/v3.yaml | 12 +-, scripts/run_pipeline.sh | 6 -, src/evaluate/base.py | 29 ++-, src/run.py | 371 ++++++++++++++++++---------, src/train/base.py | 38 +--, src/train/strategies.py | 9 +-, src/utils/data_io.py | 39 ++-, src/utils/distributed.py | 88 ++++++-, src/utils/logging.py | 9 +-, tests/integration/test_run_pipeline_modes.py | 34 ++-, tests/unit/test_trainer_evaluator.py | 28 +-, plus new files scripts/v3.sh, src/utils/losses.py, tests/e2e/artifacts/v3_hpc.yaml, tests/e2e/test_hpc_pipeline_smoke.py, tests/unit/test_losses.py
- Public API changes: `build_dataloaders(config, distributed=False, rank=0, world_size=1)`, `Evaluator(metrics, loss_config)`, `Trainer(..., loss_config=...)`, new `DistributedContext` and distributed helpers, new `LossConfig` and `binary_classification_loss`
- Config/schema changes: added `training_config.logging.validation_metrics`, added `training_config.loss.{type,pos_weight,label_smoothing}`, training CSV standardized to `Epoch,Epoch Time,Train Loss,Val Loss,Val <metric>,Learning Rate`, evaluation CSV standardized to `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`

## Decisions & Rationale
- Decision: Remove non-documented run modes and keep only `full_pipeline|train_only|eval_only`.
- Rationale: Align runtime behavior with documented contract and reduce orchestration branch complexity.
- Alternatives considered: Keep aliases for backward compatibility.
- Risk/assumption: External callers using removed modes will now fail fast with `Unsupported run mode`.

- Decision: Compute train/eval loss via shared configurable loss helper instead of model-provided `loss`.
- Rationale: Ensures consistent behavior for label smoothing and pos_weight across trainer/evaluator.
- Alternatives considered: Preserve optional model loss output and patch it post-hoc.
- Risk/assumption: Models with custom objective logic must be extended through `training_config.loss.type` in follow-up.

- Decision: Introduce distributed context lifecycle and DDP-aware orchestration.
- Rationale: Enable HPC `torchrun` execution with rank-safe logging/checkpointing and sampler sharding.
- Alternatives considered: Keep no-op distributed scaffold.
- Risk/assumption: Cluster launch environment correctly provides `RANK/LOCAL_RANK/WORLD_SIZE`.

- Decision: Standardize artifact CSV schemas with explicit field order.
- Rationale: Simplifies downstream parsing and keeps outputs stable for monitoring/reporting.
- Alternatives considered: Keep dynamic keyed columns.
- Risk/assumption: Existing parsers expecting `test_`-prefixed eval columns need update.

## State Snapshot
- Branch: scaffold
- Commit: 8a6986ca41265ee1f513457b3783fc99152aa162
- Base (optional): 2ca8255836fb6e01ce148837f67fec418b454f3c
- Uncommitted at start: yes
- Data/experiment identifiers (if any): seed=47, smoke run IDs `20260210_032155`/`20260210_032329`, HPC e2e IDs `hpc_e2e_pretrain` `hpc_e2e_finetune` `hpc_e2e_eval`

## Open Questions
- Should `scripts/run_pipeline.sh` remain deleted permanently now that `scripts/v3.sh` is the launcher?
- Should `docs/design_patterns/*.md` be updated in the same branch to match new CSV/loss/DDP behavior?
- Should distributed evaluation/test dataloaders also use `DistributedSampler` for very large-scale eval jobs?

## Next Steps (actionable, ordered)
1. Run real multi-GPU `torchrun` on target HPC node using `scripts/v3.sh` and verify rank-safe artifacts.
2. Update design docs to match implemented mode, loss, CSV, and DDP behavior.
3. Add regression tests asserting exact CSV headers/order for train and eval artifacts.
4. Decide final disposition of `scripts/run_pipeline.sh` and clean script docs accordingly.
5. Add support for additional loss types only if model roadmap requires them.

## Resume Keywords
pipeline run-modes full_pipeline train_only eval_only ddp distributed-context training_step.csv evaluate.csv loss_config pos_weight label_smoothing torchrun hpc v3.sh
