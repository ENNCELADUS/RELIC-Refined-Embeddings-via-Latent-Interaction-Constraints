# 20260209T184714Z__pipeline__strict-typing-mocked-modes

## Intent
Checkpoint strict typing stabilization and mocked run-mode integration coverage so next iteration can implement embedding loading safely without schema churn.

## Change Summary
- Files changed (short): docs/skills/usage.md, src/model/v3.py, src/run.py, src/train/base.py, src/utils/config.py, src/utils/data_io.py, tests/integration/test_run_pipeline_smoke.py (deleted), tests/integration/test_run_pipeline_modes.py (added)
- Diffstat: 7 tracked files changed with 310 insertions and 184 deletions; plus one new integration test file added.
- Public API changes: `src.run.execute_pipeline` now accepts `train_only` alias; `src.run` model construction uses factory mapping (`MODEL_FACTORIES`); config parsing helpers expanded in `src.utils.config`.
- Config/schema changes: none.

## Decisions & Rationale
- Decision: Centralize strict config coercion via `as_str/as_int/as_float/as_bool/as_str_list` and use them at pipeline boundaries.
- Rationale: Eliminates repeated object-to-primitive casts that triggered strict mypy failures.
- Alternatives considered: Scatter local `int()/float()/bool()` conversions in each module.
- Risk/assumption: Runtime `ValueError` messages must remain clear enough for debugging malformed configs.

- Decision: Keep pipeline orchestration API unchanged and add `train_only` as alias to pretrain path.
- Rationale: Supports requested mode naming while preserving existing run-mode behavior.
- Alternatives considered: Renaming mode keys globally.
- Risk/assumption: Downstream scripts may still emit `pretrain_only`; both names must be supported consistently.

- Decision: Replace PRING-coupled integration smoke test with fixture+mock orchestration tests.
- Rationale: Keeps integration coverage deterministic and independent from local dataset presence.
- Alternatives considered: Keep PRING smoke and add more tests around it.
- Risk/assumption: Mocked orchestration tests do not validate real dataloader/model side effects.

- Decision: Keep `torch.amp.GradScaler("cuda", ...)` with a targeted type-ignore for current stubs.
- Rationale: Removes runtime deprecation while retaining strict type checks.
- Alternatives considered: Keep deprecated `torch.cuda.amp.GradScaler`.
- Risk/assumption: Torch typing stubs may change and invalidate the ignore directive.

## State Snapshot
- Branch: scaffold
- Base (optional): 2ca8255836fb6e01ce148837f67fec418b454f3c
- Uncommitted at start: yes
- Data/experiment identifiers (if any): unit/integration test seed `run_config.seed=7`; mocked mode coverage: `full_pipeline`, `train_only`, `eval_only`.
- Commit: HEAD

## Open Questions
- Should `finetune_from_pretrain` and missing-checkpoint error paths get equivalent fixture+mock integration tests next?
- Should model factory mapping stay in `src/run.py` or be extracted to a dedicated registry module now that the pattern is established?
- What deterministic contract should the future real embedding loader enforce for missing cache entries and sequence length truncation?
- Is additional validation needed to guarantee scheduler fields are ignored when scheduler type is `none`?

## Next Steps (actionable, ordered)
1. Implement real embedding loading in `src/utils/data_io.py` behind existing config keys without changing schema.
2. Add integration tests for `finetune_from_pretrain` and required-checkpoint error branches using fixtures and mocks.
3. Validate end-to-end orchestration with existing scripts after embedding loader integration.
4. Decide whether to extract model registry/factory into a standalone module once a second model lands.
5. Keep strict gate check sequence (`ruff`, `mypy`, `pytest`) for each pipeline commit.

## Resume Keywords
pipeline, train_only, execute_pipeline, MODEL_FACTORIES, mypy, as_int, as_bool, GradScaler, mocked-integration, run-modes
