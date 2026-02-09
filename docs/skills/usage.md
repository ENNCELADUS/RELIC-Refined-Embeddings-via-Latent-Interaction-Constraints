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
`Use $refactor:pytorch to refactor my model + training step for readability and maintainability without changing behavior.`

7. End-to-end kickoff prompt
`Use $python-project-structure, $python-code-style, $ml-pipeline-workflow, and $python-testing-patterns to generate an MVP DL project with a runnable train/eval path and tests.`