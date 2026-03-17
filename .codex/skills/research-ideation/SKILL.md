---
name: research-ideation
description: Generate, critique, deduplicate, and rank research ideas from analyzed paper notes. Use when the user wants 3-5 evidence-grounded project ideas traced back to specific gaps, then write only the selected idea into notes/03_Projects/.
origin: Custom
---

# Research Ideation

Use this skill only after paper notes already exist.

This is not free-form brainstorming. The output should be grounded in explicit gaps, tensions, limitations, or missing evaluations found in analyzed notes.

The workflow borrows generation, critique, and ranking patterns from open-coscientist, plus gap-lineage tracking inspired by MLEvolve and RD-Agent.

## When To Activate

- the user has 2-6 analyzed paper notes
- the user wants research directions grounded in those notes
- the user wants ranked ideas, not just a loose brainstorm

## Do Not Use For

- raw literature search
- single-paper analysis
- code generation
- experiment execution

## Inputs

- 2-6 source notes under `notes/01_Papers/`, or one deep-research note plus related paper notes
- optional constraints such as subdomain, resource limits, evaluation setting, risk tolerance, or timeline

## Workflow

1. Read the source notes, not just the titles.
2. Extract the most important unresolved gaps, assumptions, contradictions, and missing evaluations.
3. Cluster overlapping gaps into 2-4 opportunity areas.
4. Generate 3-5 candidate ideas. Each idea must combine a concrete gap with a plausible research move or study design.
5. For each idea, state:
   - one-sentence thesis
   - gap lineage
   - why it might matter
   - what could falsify it quickly
   - what makes it different from the source papers
6. Run an internal critique pass:
   - remove near-duplicates
   - downgrade vague ideas with no tractable validation path
   - reject ideas that are only `bigger model`, `more data`, or `apply method X to domain Y` without a sharper hypothesis
7. Rank the remaining ideas by novelty, evidence fit, and tractability.
8. Present the shortlist first.
9. Only create a note after the user selects one or two ideas.
10. When creating a note, write it under `notes/03_Projects/Idea - <Short Name>.md`.

## Output Before Selection

```markdown
## Idea Shortlist

### Idea 1: ...
- One-sentence thesis:
- Gap lineage:
- Why it might matter:
- Fastest falsification test:
- Main risk:

### Idea 2: ...
- One-sentence thesis:
- Gap lineage:
- Why it might matter:
- Fastest falsification test:
- Main risk:

## Ranking
| Rank | Idea | Novelty | Evidence Fit | Tractability | Why Now |
| --- | --- | --- | --- | --- | --- |
```

## Note Template After Selection

```markdown
# Idea: [Name]
[[Source Note]]
[[Source Note]]

## Selected Idea
A concise statement of the proposed research idea.

## Gap Lineage
- Source notes:
- Gaps combined:
- Why this is not already solved:

## Core Contribution & Context
Short context grounded in the analyzed notes.

## Validation Path
- First experiment or analysis:
- Required data / resources:
- Main failure condition:

## Risks / Falsifiers
- ...
- ...

## References
- [[Source Note]]
- [[Source Note]]
```

## Quality Rules

- Ground every idea in explicit evidence from the input notes.
- Separate observation from speculation.
- Prefer specific hypotheses over vague performance-improvement claims.
- Avoid duplicate ideas with different wording.
- Do not create a project note until the user selects an idea unless the user explicitly asks for automatic note creation.

## Out Of Scope

- finding papers from scratch
- summarizing a single paper
- implementing code or running experiments
