---
name: paper-analysis
description: Analyze a single research paper already stored in literature/ and turn it into a gap-focused Obsidian note. Use when the user wants per-paper analysis with contributions, assumptions, limitations, failure modes, labeled inference, and [[wikilink]] connections.
origin: Custom
---

# Paper Analysis

Use this skill after the paper has already been found and imported into the vault.

The goal is not generic summarization. The goal is to produce a reading note that is useful for later comparison, gap finding, and idea generation.

This workflow blends vault note-taking with a verification pass inspired by research-agent systems such as InternAgent and open-coscientist.

## When To Activate

- a single paper in `literature/` needs to be analyzed
- the user wants a note in `notes/01_Papers/`
- the user wants gaps, assumptions, or failure modes from one paper

## Do Not Use For

- broad literature search across many papers
- ranking papers for download
- generating project ideas from multiple notes
- code implementation or experiment reproduction

## Inputs

- a PDF path under `literature/`, or a paper title that can be resolved to one file
- optional related note names, concepts, or benchmarks to connect

## Required Extraction Tools

- Always read the paper with both text and image extraction
- Use `pdftotext` to extract the paper text so the note is grounded in the exact wording, tables, and section structure
- Use `pdftocairo` to render selected pages with figures, plots, diagrams, or result tables so the model can inspect the visual evidence directly
- Do not rely on only one modality when the paper contains important plots, architecture diagrams, qualitative examples, or figure-only ablations

## Naming Standard

- Keep the note in the best-fit subfolder under `notes/01_Papers/`
- Mirror the paper category when it is clear, for example:
  - `literature/benchmarks/...pdf` -> `notes/01_Papers/benchmarks/...md`
  - `literature/networks/...pdf` -> `notes/01_Papers/networks/...md`
- The note filename must exactly match the source PDF basename, replacing `.pdf` with `.md`
- Example:
  - source PDF: `literature/networks/2025_nature_plm_interact_extending_protein_language_models_predict_protein.pdf`
  - note file: `notes/01_Papers/networks/2025_nature_plm_interact_extending_protein_language_models_predict_protein.md`
- Keep the human-readable paper title in the note body as `# Title`
- When linking to another paper note, prefer Obsidian alias syntax so links stay readable:
  - `[[2025_arxiv_pring_rethinking_protein_protein_interaction_prediction_pairs_gr|PRING: Rethinking Protein-Protein Interaction Prediction from Pairs to Graphs]]`

## Workflow

1. Resolve the paper file, capture its exact basename, and check whether a note for the same PDF basename already exists.
2. If a note already exists, update it instead of creating a duplicate.
3. Extract full paper text with `pdftotext` and read the resulting text before drafting anything.
4. Identify the most visually important pages, then render them with `pdftocairo` so figures, plots, architecture diagrams, and table layouts can be inspected.
5. Read for problem setup, method, evaluation design, strongest evidence, and scope of claims across both the extracted text and the rendered visuals.
6. Separate paper-stated facts from your own inference. If a limitation or risk is inferred, label it clearly.
7. Extract the core contribution in concise prose and short bullets.
8. Extract the most decision-relevant experimental details: task, data, baselines, metrics, and evaluation regime.
9. Identify assumptions, technical limitations, missing evaluations, and likely edge cases or brittle settings.
10. Search `notes/` for related concepts or papers and add targeted `[[NoteName]]` links when the connection is real. For paper notes, use the PDF-basename note id with a readable alias when needed.
11. Write the note under the best-fit location in `notes/01_Papers/`:
   - use an existing topical subfolder such as `survey/`, `benchmarks/`, `networks/`, or `ml_tricks/` when it clearly fits
   - otherwise write directly under `notes/01_Papers/`
   - the final note filename must still match the source PDF basename with `.md`

## Recommended Commands

Extract text:

```bash
pdftotext -layout literature/<paper>.pdf -
```

Render selected pages to PNG for figure reading:

```bash
pdftocairo -png -f <start_page> -l <end_page> literature/<paper>.pdf /tmp/paper_fig
```

Use a small page range first, then expand only if needed. Prioritize pages with model diagrams, benchmark plots, ablations, and qualitative examples.

## Output Template

```markdown
# Title
**Authors**: ...
**Year**: ...
**Venue**: ...

## Summary
2-3 sentences in your own words.

## Methods
- Task setup:
- Model or method:
- Data / benchmarks:
- Evaluation:

## Results & Key Findings
- ...
- ...

## Gaps & Limitations
### Explicit in Paper
- ...

### Inference
- ...

## Edge Cases / Failure Modes
- ...
- ...

## My Thoughts / Questions
- ...
- ...

## Links to Concepts / Other Papers
- [[ConceptOrPaper]]
- [[ConceptOrPaper]]
```

## Quality Rules

- Do not just paraphrase the abstract.
- Do not invent numbers, claims, datasets, or baselines.
- Prefer short evidence-backed bullets over long prose.
- If information is missing, say `Not clearly reported` instead of guessing.
- If the paper is a survey or benchmark paper, adapt `Methods` and `Results & Key Findings` to match that genre instead of forcing a method-paper format.
- Do not invent a custom note filename. The note file must stay aligned with the source PDF basename.
- Do not skip figure reading when plots or diagrams carry key evidence that is weakly described in the text.

## Out Of Scope

- literature retrieval
- novelty scanning across a whole field
- project idea generation
- code generation or benchmark automation
