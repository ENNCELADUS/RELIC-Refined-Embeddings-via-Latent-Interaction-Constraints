---
name: deep-research-ingest
description: Read a Deep Research markdown report, extract paper titles, years, and PDF URLs, download the PDFs, and save them into literature/ as year_source_short_name.pdf. Use when the user has already completed a Deep Research pass and wants local PDF ingestion from the report.
origin: Custom
---

# Deep Research Ingest

Use this skill after a Deep Research report already exists.

This skill closes the gap between a retrieval report and the local PDF library. It is for ingestion only: read the report, extract candidates, download PDFs, and save them with stable filenames.

## When To Activate

- a report such as `notes/03_Projects/deep-research-report.md` already exists
- the report contains paper titles, years, and direct PDF URLs or equivalent
- the user wants local PDF files under `literature/`

## Do Not Use For

- writing paper notes
- generating research ideas
- re-running literature search
- fixing incorrect paper metadata by hand unless download parsing fails

## Script

- Primary tool: `.codex/skills/deep-research-ingest/scripts/download_papers_from_report.py`
- The parser supports:
  - the current bold-field report format with lines such as `**Full title:**`, `**Year:**`, and `**Direct PDF URL:**`
  - the table-style Deep Research template that uses `Paper`, `Year`, and `PDF Route` columns

## Naming Rule

Every downloaded PDF is saved as:

```text
year_source_short_name.pdf
```

Where:
- `year` comes from the report
- `source` comes from the PDF URL host, normalized to a short stable label such as `openreview`, `biorxiv`, `nature`, `arxiv`, or `oup`
- `short_name` comes from a normalized, truncated title slug

Files are saved directly into `literature/`.

## Recommended Workflow

1. Inspect the report path and make sure it is the intended report.
2. Run a dry run first.
3. Review skipped items, especially entries with missing or non-direct PDF URLs.
4. Run the real download command.
5. Keep existing files unless overwrite is explicitly requested.
6. Move to `paper-analysis` only after the PDF is local.

## Commands

Preview the top queue:

```bash
python .codex/skills/deep-research-ingest/scripts/download_papers_from_report.py \
  --report notes/03_Projects/deep-research-report.md \
  --section top \
  --dry-run
```

Download the top queue:

```bash
python .codex/skills/deep-research-ingest/scripts/download_papers_from_report.py \
  --report notes/03_Projects/deep-research-report.md \
  --section top
```

Download all parsed entries:

```bash
python .codex/skills/deep-research-ingest/scripts/download_papers_from_report.py \
  --report notes/03_Projects/deep-research-report.md \
  --section all
```

Overwrite existing files if needed:

```bash
python .codex/skills/deep-research-ingest/scripts/download_papers_from_report.py \
  --report notes/03_Projects/deep-research-report.md \
  --section top \
  --overwrite
```

## Quality Rules

- Prefer the report's direct PDF URL when available.
- If the report lacks a direct PDF URL but the abstract URL allows a safe host-specific PDF guess, the script may use that fallback.
- If a response does not look like a PDF, skip it and report the failure instead of saving HTML as a `.pdf`.
- Do not rename an existing local file unless overwrite is explicit.
- Keep the workflow deterministic: parse, preview, download, then hand off to analysis.

## Out Of Scope

- per-paper note writing
- idea generation
- citation formatting
- metadata enrichment beyond what is needed for filename generation and download
