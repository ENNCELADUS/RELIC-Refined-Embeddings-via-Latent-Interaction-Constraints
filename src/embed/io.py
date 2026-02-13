"""Embedding sequence discovery and parsing helpers."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path

from src.embed.config import (
    CSV_SUFFIXES,
    DEFAULT_ID_COLUMNS,
    DEFAULT_SEQUENCE_COLUMNS,
    FASTA_SUFFIXES,
    FASTA_UNIPROT_PATTERN,
    _optional_string,
)
from src.utils.config import ConfigDict, get_section


def _collect_required_protein_ids(
    split_paths: Sequence[Path],
) -> set[str]:
    """Collect required protein IDs from configured split files."""
    required_ids: set[str] = set()
    for split_path in split_paths:
        if not split_path.exists():
            raise FileNotFoundError(f"Split dataset path not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [part.strip() for part in line.strip().split("\t")]
                if len(parts) < 2:
                    continue
                if not parts[0] or not parts[1]:
                    continue

                required_ids.add(parts[0])
                required_ids.add(parts[1])

    if not required_ids:
        raise ValueError("No protein IDs found in configured split files")
    return required_ids


def _resolve_sequence_search_roots(config: ConfigDict, split_paths: Sequence[Path]) -> list[Path]:
    """Build ordered sequence-discovery roots from config and split paths."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")

    candidate_roots: list[Path] = []
    seen: set[str] = set()

    def _add_if_new(path: Path | None) -> None:
        if path is None:
            return
        normalized = str(path.resolve())
        if normalized in seen:
            return
        seen.add(normalized)
        candidate_roots.append(path)

    root_dir_raw = _optional_string(benchmark_cfg.get("root_dir"))
    processed_dir_raw = _optional_string(benchmark_cfg.get("processed_dir"))
    _add_if_new(Path(processed_dir_raw) if processed_dir_raw is not None else None)
    _add_if_new(Path(root_dir_raw) if root_dir_raw is not None else None)

    for split_path in split_paths:
        _add_if_new(split_path.parent)
        _add_if_new(split_path.parent.parent)

    return [path for path in candidate_roots if path.exists()]


def _collect_candidate_files(search_roots: Sequence[Path], suffixes: Sequence[str]) -> list[Path]:
    """Collect candidate files under search roots for the specified suffixes."""
    candidates: list[Path] = []
    seen: set[str] = set()

    for root in search_roots:
        for suffix in suffixes:
            for path in root.rglob(f"*{suffix}"):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(path)

    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates


def _clean_protein_sequence(sequence: str) -> str:
    """Canonicalize a protein sequence string."""
    cleaned = sequence.strip().upper()
    cleaned = cleaned.replace("-", "").replace(".", "").replace("*", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _resolve_column_name(
    fieldnames: Sequence[str],
    override: str | None,
    candidates: Sequence[str],
    column_kind: str,
    csv_path: Path,
) -> str | None:
    """Resolve a CSV column name by override or known candidates."""
    lowercase_to_actual = {name.lower(): name for name in fieldnames}
    if override is not None:
        resolved = lowercase_to_actual.get(override.lower())
        if resolved is None:
            raise ValueError(
                f"CSV file {csv_path} missing requested {column_kind} column '{override}'"
            )
        return resolved

    for candidate in candidates:
        resolved = lowercase_to_actual.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def _load_sequences_from_csv(
    csv_path: Path,
    required_ids: set[str],
    sequences: dict[str, str],
    id_column_override: str | None,
    sequence_column_override: str | None,
) -> None:
    """Load matching protein sequences from a CSV file into ``sequences``."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return

        id_column = _resolve_column_name(
            fieldnames=reader.fieldnames,
            override=id_column_override,
            candidates=DEFAULT_ID_COLUMNS,
            column_kind="ID",
            csv_path=csv_path,
        )
        sequence_column = _resolve_column_name(
            fieldnames=reader.fieldnames,
            override=sequence_column_override,
            candidates=DEFAULT_SEQUENCE_COLUMNS,
            column_kind="sequence",
            csv_path=csv_path,
        )
        if id_column is None or sequence_column is None:
            return

        for row in reader:
            protein_id = row.get(id_column, "").strip()
            if protein_id not in required_ids or protein_id in sequences:
                continue
            sequence_value = _clean_protein_sequence(row.get(sequence_column, ""))
            if sequence_value:
                sequences[protein_id] = sequence_value
            if len(sequences) >= len(required_ids):
                return


def _extract_protein_id_from_header(header: str) -> str:
    """Extract protein ID from FASTA header."""
    match = FASTA_UNIPROT_PATTERN.search(header)
    if match is not None:
        return match.group(1).strip()
    token = header.split(maxsplit=1)[0]
    return token.strip()


def _load_sequences_from_fasta(
    fasta_path: Path,
    required_ids: set[str],
    sequences: dict[str, str],
) -> None:
    """Load matching protein sequences from a FASTA file into ``sequences``."""
    current_id: str | None = None
    current_chunks: list[str] = []

    def _flush_current() -> None:
        nonlocal current_id
        nonlocal current_chunks
        if current_id is None or current_id in sequences or current_id not in required_ids:
            current_id = None
            current_chunks = []
            return
        sequence_value = _clean_protein_sequence("".join(current_chunks))
        if sequence_value:
            sequences[current_id] = sequence_value
        current_id = None
        current_chunks = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                _flush_current()
                current_id = _extract_protein_id_from_header(stripped[1:])
                continue
            if current_id is not None:
                current_chunks.append(stripped)
        _flush_current()


def _discover_sequences(
    required_ids: set[str],
    search_roots: Sequence[Path],
    explicit_sequence_file: Path | None,
    id_column_override: str | None,
    sequence_column_override: str | None,
) -> dict[str, str]:
    """Discover protein sequences using CSV-first then FASTA fallback."""
    sequences: dict[str, str] = {}

    def _load_from_path(path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in CSV_SUFFIXES:
            _load_sequences_from_csv(
                csv_path=path,
                required_ids=required_ids,
                sequences=sequences,
                id_column_override=id_column_override,
                sequence_column_override=sequence_column_override,
            )
            return
        if suffix in FASTA_SUFFIXES:
            _load_sequences_from_fasta(
                fasta_path=path,
                required_ids=required_ids,
                sequences=sequences,
            )
            return
        raise ValueError(f"Unsupported sequence_file extension: {path}")

    if explicit_sequence_file is not None:
        if not explicit_sequence_file.exists():
            raise FileNotFoundError(f"Configured sequence file not found: {explicit_sequence_file}")
        _load_from_path(explicit_sequence_file)

    if len(sequences) < len(required_ids):
        csv_files = _collect_candidate_files(search_roots=search_roots, suffixes=CSV_SUFFIXES)
        for csv_path in csv_files:
            if (
                explicit_sequence_file is not None
                and csv_path.resolve() == explicit_sequence_file.resolve()
            ):
                continue
            _load_sequences_from_csv(
                csv_path=csv_path,
                required_ids=required_ids,
                sequences=sequences,
                id_column_override=id_column_override,
                sequence_column_override=sequence_column_override,
            )
            if len(sequences) >= len(required_ids):
                break

    if len(sequences) < len(required_ids):
        fasta_files = _collect_candidate_files(search_roots=search_roots, suffixes=FASTA_SUFFIXES)
        for fasta_path in fasta_files:
            if (
                explicit_sequence_file is not None
                and fasta_path.resolve() == explicit_sequence_file.resolve()
            ):
                continue
            _load_sequences_from_fasta(
                fasta_path=fasta_path,
                required_ids=required_ids,
                sequences=sequences,
            )
            if len(sequences) >= len(required_ids):
                break

    missing_ids = sorted(required_ids - set(sequences))
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise FileNotFoundError(
            "Unable to discover sequences for required proteins: "
            f"{preview} (missing={len(missing_ids)})"
        )
    return sequences
