# 20260209T180037Z__scaffold__generic-pipeline-scaffold

## Intent
Checkpoint scaffold and code-style unification changes so the next iteration can resume from a generic trainer/evaluator pipeline baseline with strict lint/type/doc standards.

## Change Summary
- Files changed (short): AGENTS.md, CLAUDE.md, configs/v3.yaml, pyproject.toml, src/model/v3.py, src/utils/data_samplers.py, docs/skills/usage.md, src/__init__.py, src/run.py, src/train/*, src/evaluate/*, src/utils/{config,data_io,device,distributed,early_stop,logging,ohem_sample_strategy}.py, tests/*
- Diffstat: 6 tracked files currently show 269 insertions / 202 deletions pre-stage; plus scaffold and tests as untracked additions.
- Public API changes: `src.run.execute_pipeline`, `src.train.base.Trainer`, `src.evaluate.base.Evaluator`, `src.utils.ohem_sample_strategy.{select_ohem_indices,OHEMSampleStrategy}`, package exports in `src/__init__.py` and subpackage `__init__` files.
- Config/schema changes: expanded `configs/v3.yaml` schema for stage-based run modes, benchmark block (`data_config.benchmark.*`), dataloader sampling (`strategy/keep_ratio/min_keep`), and trainer strategy config.

## Decisions & Rationale
- Decision: Use a centralized `src/run.py` orchestrator with generic `Trainer` and `Evaluator` contracts.
- Rationale: Matches `docs/design_patterns/*` requirements and keeps model onboarding simple.
- Alternatives considered: Per-model bespoke train/eval scripts.
- Risk/assumption: Assumes every model can satisfy the forward contract returning `logits` and optional `loss`.

- Decision: Add OHEM as a utility strategy (`src/utils/ohem_sample_strategy.py`) and keep data samplers separate.
- Rationale: Separates hard-example policy from trainer loop mechanics.
- Alternatives considered: Embedding OHEM logic directly in trainer.
- Risk/assumption: Assumes per-sample loss is available or derivable from logits+labels.

- Decision: Enforce strict style gates through `ruff` + `mypy` configuration in `pyproject.toml`.
- Rationale: Enforces no-print, type hints, and Google-style docs consistently.
- Alternatives considered: Soft style guidance without automated checks.
- Risk/assumption: Assumes development env includes `ruff` and `mypy`.

## State Snapshot
- Branch: scaffold
- Base (optional): 2ca8255836fb6e01ce148837f67fec418b454f3c
- Uncommitted at start: yes
- Data/experiment identifiers (if any): benchmark=PRING, seed=47, default split=human/BFS
- Commit: HEAD

## Open Questions
- Should AGENTS.md and CLAUDE.md changes be retained in this scaffold checkpoint or split into a separate helper checkpoint?
- Should `docs/skills/usage.md` be part of scaffold scope or moved to documentation scope?
- Should synthetic embedding generation in `src/utils/data_io.py` be replaced with ESM embedding loading in the next commit?

## Next Steps (actionable, ordered)
1. Install `mypy` in the `relic` environment and run strict type checks on `src` and `tests`.
2. Add model registry/factory wiring for future models beyond `v3` without expanding if/elif chains.
3. Replace synthetic PRING embeddings with deterministic loading from cached ESM embeddings.
4. Add integration tests for `pretrain_only`, `finetune_from_pretrain`, and `eval_only` run modes.
5. Decide whether AGENTS/CLAUDE and docs/skills files belong in scaffold scope and split if needed.

## Resume Keywords
scaffold, run.py, Trainer, Evaluator, OHEM, PRING, v3, config-schema, staged_unfreeze, ruff, mypy, google-docstring
