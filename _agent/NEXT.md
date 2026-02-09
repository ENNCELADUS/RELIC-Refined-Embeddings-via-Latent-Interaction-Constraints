# NEXT

## Read This First
- Current focus: stabilize scaffold baseline and close remaining strict type-checking gaps.
- Success criteria for the *next* commit: `mypy src tests` passes in `esm`, and pipeline modes have explicit integration coverage.

## Where to Resume
- Helper-Checkpoint: 20260209T180037Z__scaffold__generic-pipeline-scaffold
- Helper-Note: _agent/notes/20260209T180037Z__scaffold__generic-pipeline-scaffold.md
- Key files touched: src/run.py, src/train/base.py, src/evaluate/base.py, src/model/v3.py, src/utils/*.py, configs/v3.yaml, pyproject.toml, tests/*

## Next Commit Plan (strict order)
1. Install and execute mypy in `esm`; fix strict typing errors in `src/model/v3.py` and pipeline modules.
2. Add/clean model-factory abstraction in `src/run.py` for generic multi-model support.
3. Implement real embedding loader path in `src/utils/data_io.py` behind config toggle.
4. Add tests for `pretrain_only`, `finetune_from_pretrain`, and `eval_only`.
5. Separate non-scaffold docs changes if they should not remain in this scope.

## Guardrails
- What must not change: model forward contract (`logits` + optional `loss`), centralized run-mode orchestration, strict no-print policy.
- What should be refactored later (explicitly deferred): replace if/elif model selection with registry module and split oversized model internals into focused submodules.

## Quick Retrieval
Commands the next iteration should run first:
- `git log --grep "^helper(" -n 30`
- `rg "scaffold|generic-pipeline-scaffold|execute_pipeline|OHEMSampleStrategy" _agent`
