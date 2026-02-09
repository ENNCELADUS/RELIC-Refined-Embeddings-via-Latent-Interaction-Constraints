# NEXT

## Read This First
- Current focus: implement real embedding loading path in `src/utils/data_io.py` while preserving strict typing and mocked orchestration coverage.
- Success criteria for the *next* commit: embedding loader path is implemented without schema changes; `mypy src tests` and `python -m pytest` both pass; fixture+mock integration coverage includes `finetune_from_pretrain` and checkpoint error paths.

## Where to Resume
- Helper-Checkpoint: 20260209T184714Z__pipeline__strict-typing-mocked-modes
- Helper-Note: _agent/notes/20260209T184714Z__pipeline__strict-typing-mocked-modes.md
- Key files touched: src/run.py, src/utils/config.py, src/train/base.py, src/model/v3.py, src/utils/data_io.py, tests/integration/test_run_pipeline_modes.py

## Next Commit Plan (strict order)
1. Implement deterministic cached embedding loader logic in `src/utils/data_io.py` using existing config keys only.
2. Keep `src/run.py` orchestration contracts stable (`execute_pipeline`, mode dispatch, forward contract assumptions).
3. Add fixture+mock integration tests for `finetune_from_pretrain` and load-checkpoint validation failures in `tests/integration`.
4. Extend/adjust unit tests only where embedding loader behavior requires new edge-case validation.
5. Run `ruff check --fix .`, `ruff format .`, `mypy src tests`, and `python -m pytest` and resolve all regressions.

## Guardrails
- What must not change: model forward contract (`logits` + optional `loss`), centralized run-mode orchestration, strict no-print policy, and current config schema.
- What should be refactored later (explicitly deferred): extracting model registry into its own module and splitting large `src/model/v3.py` internals.

## Quick Retrieval
Commands the next iteration should run first:
- `git log --grep "^helper(" -n 30`
- `rg "pipeline|strict-typing-mocked-modes|train_only|MODEL_FACTORIES" _agent`
