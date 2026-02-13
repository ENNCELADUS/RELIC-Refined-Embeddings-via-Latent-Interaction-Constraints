1. Bootstrap project structure  
`Use $python-project-structure to scaffold a new PyTorch DL repo for protein sequence classification with src/, configs/, scripts/, and tests/.`

2. Define style + linting baseline  
`Use $python-code-style to set up ruff, strict type hints, logging (no print), and Google-style docstrings across the new project.`

Loop:
3. Build training pipeline skeleton

每新增一个 pipeline stage、一个 Trainer 能力（AMP、grad accum、DDP hooks）、或一类 metric/ckpt 行为，就走这轮。

`Use $resume-from-helper, then follow the plan exactly.`
`Use $ml-pipeline-workflow to create a config-driven pipeline with preprocess, train, evaluate stages and a Trainer class.`
`Then use $helper-checkpoint with scope=pipeline slug=trainer-stage-<feature>.`

建议 slug 例子：trainer-stage-dataloader、trainer-stage-metrics、trainer-stage-checkpointing。

建议顺序：config → dataloader → trainer step → evaluate → CLI/scripts。

4. Add robust tests early

每当 Loop 3 加了能力，就立刻补测试，避免后续 refactor 失控。

`Use $python-testing-patterns to write pytest unit + integration tests for config parsing, dataloaders, and trainer loops.`
`Then use $helper-checkpoint with scope=tests slug=cover-<module>.`

5. Review and improve code quality  
`Use $python-code-quality to run a quality pass on src/train and src/model and fix complexity, typing, and exception handling issues.`

6. Refactor PyTorch model/training code

## A) PyTorch-specific skills (`refactor:pytorch`, `debug:pytorch`)

### 1) `refactor:pytorch` — refactor a big training file (behavior-preserving)

```text
Use $refactor:pytorch to refactor src/train/trainer.py (≈900 LOC) into maintainable modules WITHOUT changing behavior.

Constraints:
- Preserve public APIs and config keys.
- Keep training semantics identical (same outputs for same seed/config).
- Split responsibilities: trainer core, loss/metrics, logging, checkpointing, eval loop.
- Prefer PyTorch 2.x patterns (torch.compile where appropriate; AMP structure; dataloader best practices).
- After refactor: run `pytest -q` and a 1-epoch smoke run.
Deliver:
- A short “moved where” map (old symbol -> new module).
```

---

## B) `mcp-python-refactoring` (MCP server) — analyze & get extraction guidance

This MCP server exposes tools like `analyze_python_file`, `analyze_python_package`, `find_long_functions`, `get_extraction_guidance`, and `tdd_refactoring_guidance`. ([Docker Hub][4])

### Example 1 — package scan to find “god modules” / structural issues

```text
Use the mcp-python-refactoring tool `analyze_python_package` with:
- package_path = "src/train"

Return:
- Top 10 refactor opportunities with file paths + severity
- Metrics summary (complexity hotspots)
- 3 suggested module-boundary changes (what to split/merge)
```

### Example 2 — target a single file and request “extract method” line-level guidance

```text
Use the mcp-python-refactoring tool `analyze_python_file` on "src/train/trainer.py".
Then run `find_long_functions` (threshold 40 lines) on the same content.
For the worst offender, call `get_extraction_guidance` and produce an ordered extraction plan.
```

### Example 3 — TDD-style refactor plan (test first → refactor → test)

```text
Use the mcp-python-refactoring tool `tdd_refactoring_guidance` on the function `train_one_epoch`
and target tests under "tests/".
Ask for:
- characterization tests to lock behavior
- refactor sequence (micro-steps)
- post-refactor checks
```

---

## C) `refactoring-surgeon` — general, behavior-preserving cleanup (tech debt)

This skill is explicitly “improve code quality without changing behavior” and emphasizes small steps + tests.

### Example — de-duplicate utilities, reduce complexity, keep behavior identical

```text
Use $refactoring-surgeon to refactor src/ without changing behavior.

Scope:
- Eliminate duplication across metric functions
- Reduce conditional complexity
- Improve naming and docstrings
- Keep signatures stable

Safety:
- Add/extend characterization tests in tests/ first
- Refactor in small commits; run `pytest -q` after each change

Deliver:
- A list of code smells found (by name)
- The exact refactor techniques applied (Extract Method, Introduce Parameter Object, etc.)
```

7. End-to-end kickoff prompt
`Use $python-project-structure, $python-code-style, $ml-pipeline-workflow, and $python-testing-patterns to generate an MVP DL project with a runnable train/eval path and tests.`