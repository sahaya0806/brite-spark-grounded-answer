"""
Unit and integration tests for Milestone 2 — Markdown Policy Ingestion.

Test groups
-----------
1. Loader — happy path (valid document)
2. Loader — error paths (missing, directory, empty, non-UTF-8)
3. Inspection — headings
4. Inspection — lists and tables
5. Inspection — clause IDs and cross-references
6. Inspection — immutability / no source mutation
7. Real corpus integration test
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.ingestion.loader import (
    PolicyDocument,
    PolicyLoadError,
    load_policy_document,
)
from src.ingestion.inspector import (
    MarkdownInspection,
    inspect_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REAL_CORPUS = Path("data/raw/policy_manual.md")


def _write(tmp_path: Path, filename: str, content: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding=encoding)
    return p


def _write_bytes(tmp_path: Path, filename: str, data: bytes) -> Path:
    p = tmp_path / filename
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# 1. Loader — happy path
# ---------------------------------------------------------------------------

class TestLoaderHappyPath:

    def test_returns_policy_document(self, tmp_path: Path):
        p = _write(tmp_path, "doc.md", "# Hello\n\nSome text.")
        doc = load_policy_document(p)
        assert isinstance(doc, PolicyDocument)

    def test_source_path_is_absolute(self, tmp_path: Path):
        p = _write(tmp_path, "doc.md", "# Hello\n\nSome text.")
        doc = load_policy_document(p)
        assert doc.source_path.is_absolute()

    def test_raw_text_matches_source_exactly(self, tmp_path: Path):
        content = "# Title\n\n**1.1** First clause.\n"
        p = _write(tmp_path, "doc.md", content)
        doc = load_policy_document(p)
        assert doc.raw_text == content

    def test_character_count_is_correct(self, tmp_path: Path):
        content = "# Title\n\nHello world.\n"
        p = _write(tmp_path, "doc.md", content)
        doc = load_policy_document(p)
        assert doc.character_count == len(content)

    def test_line_count_is_correct(self, tmp_path: Path):
        content = "line one\nline two\nline three"
        p = _write(tmp_path, "doc.md", content)
        doc = load_policy_document(p)
        assert doc.line_count == len(content.splitlines())

    def test_utf8_content_preserved(self, tmp_path: Path):
        content = "# Résumé\n\n€500 per month — full stop.\n"
        p = _write(tmp_path, "doc.md", content, encoding="utf-8")
        doc = load_policy_document(p)
        assert "Résumé" in doc.raw_text
        assert "€500" in doc.raw_text

    def test_accepts_path_as_string(self, tmp_path: Path):
        p = _write(tmp_path, "doc.md", "# Hello\n\nText.")
        doc = load_policy_document(str(p))
        assert isinstance(doc, PolicyDocument)

    def test_document_is_immutable(self, tmp_path: Path):
        p = _write(tmp_path, "doc.md", "# Hello\n\nText.")
        doc = load_policy_document(p)
        with pytest.raises((AttributeError, TypeError)):
            doc.raw_text = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Loader — error paths
# ---------------------------------------------------------------------------

class TestLoaderErrors:

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(PolicyLoadError, match="not found"):
            load_policy_document(tmp_path / "does_not_exist.md")

    def test_directory_path_raises(self, tmp_path: Path):
        with pytest.raises(PolicyLoadError, match="not a file"):
            load_policy_document(tmp_path)

    def test_empty_file_raises(self, tmp_path: Path):
        p = _write(tmp_path, "empty.md", "")
        with pytest.raises(PolicyLoadError, match="empty"):
            load_policy_document(p)

    def test_whitespace_only_file_raises(self, tmp_path: Path):
        p = _write(tmp_path, "blank.md", "   \n\n   \n")
        with pytest.raises(PolicyLoadError, match="empty"):
            load_policy_document(p)

    def test_non_utf8_file_raises(self, tmp_path: Path):
        # Write bytes that are not valid UTF-8
        p = _write_bytes(tmp_path, "bad.md", b"Hello \xff\xfe world")
        with pytest.raises(PolicyLoadError):
            load_policy_document(p)


# ---------------------------------------------------------------------------
# 3. Inspection — headings
# ---------------------------------------------------------------------------

SIMPLE_DOC = """\
# Part 1 — Introduction

## 1.1 Purpose

**1.1.1** First clause.

## 1.2 Scope

**1.2.1** Second clause.

# Part 2 — Eligibility

## 2.1 Basic conditions

**2.1.1** Third clause.
"""


def _load(tmp_path: Path, content: str) -> PolicyDocument:
    p = _write(tmp_path, "doc.md", content)
    return load_policy_document(p)


class TestInspectionHeadings:

    def test_heading_count(self, tmp_path: Path):
        doc = _load(tmp_path, SIMPLE_DOC)
        insp = inspect_markdown(doc)
        assert len(insp.headings) == 5  # 2× H1, 3× H2

    def test_heading_levels(self, tmp_path: Path):
        doc = _load(tmp_path, SIMPLE_DOC)
        insp = inspect_markdown(doc)
        levels = [h.level for h in insp.headings]
        assert levels == [1, 2, 2, 1, 2]

    def test_heading_text(self, tmp_path: Path):
        doc = _load(tmp_path, SIMPLE_DOC)
        insp = inspect_markdown(doc)
        assert insp.headings[0].text == "Part 1 — Introduction"
        assert insp.headings[1].text == "1.1 Purpose"

    def test_heading_counts_by_level(self, tmp_path: Path):
        doc = _load(tmp_path, SIMPLE_DOC)
        insp = inspect_markdown(doc)
        assert insp.heading_counts_by_level[1] == 2
        assert insp.heading_counts_by_level[2] == 3

    def test_heading_line_numbers_ascending(self, tmp_path: Path):
        doc = _load(tmp_path, SIMPLE_DOC)
        insp = inspect_markdown(doc)
        line_nums = [h.line_number for h in insp.headings]
        assert line_nums == sorted(line_nums)

    def test_no_headings_document(self, tmp_path: Path):
        doc = _load(tmp_path, "Just some plain text with no headings.\n")
        insp = inspect_markdown(doc)
        assert len(insp.headings) == 0
        assert insp.heading_counts_by_level == {}


# ---------------------------------------------------------------------------
# 4. Inspection — lists and tables
# ---------------------------------------------------------------------------

LIST_TABLE_DOC = """\
# Part 1

## 1.1 Lists

The following apply —

(a) first item;

(b) second item;

(c) third item.

## 1.2 Table

| Column A | Column B |
|:--|:--|
| row 1a   | row 1b   |
| row 2a   | row 2b   |
"""


class TestInspectionListsAndTables:

    def test_lettered_items_counted(self, tmp_path: Path):
        doc = _load(tmp_path, LIST_TABLE_DOC)
        insp = inspect_markdown(doc)
        assert insp.ordered_list_item_count == 3

    def test_table_rows_counted(self, tmp_path: Path):
        doc = _load(tmp_path, LIST_TABLE_DOC)
        insp = inspect_markdown(doc)
        # header + separator + 2 data rows = 4 table rows
        assert insp.table_row_count == 4

    def test_unordered_items_zero_when_absent(self, tmp_path: Path):
        doc = _load(tmp_path, LIST_TABLE_DOC)
        insp = inspect_markdown(doc)
        assert insp.unordered_list_item_count == 0

    def test_unordered_items_detected(self, tmp_path: Path):
        content = "# Section\n\n- apple\n- banana\n- cherry\n"
        doc = _load(tmp_path, content)
        insp = inspect_markdown(doc)
        assert insp.unordered_list_item_count == 3


# ---------------------------------------------------------------------------
# 5. Inspection — clause IDs and cross-references
# ---------------------------------------------------------------------------

CLAUSE_DOC = """\
# Part 4 — Obligations

## 4.3 Recipient obligations

**4.3.1** A recipient must comply with §2.1.2 and §4.3.2.

**4.3.2** Report within 10 days as required under §4.3.

**4.3.3** Failure may result in §10.5.
"""


class TestInspectionClauseIds:

    def test_possible_clause_ids_found(self, tmp_path: Path):
        doc = _load(tmp_path, CLAUSE_DOC)
        insp = inspect_markdown(doc)
        assert "4.3.1" in insp.possible_clause_ids
        assert "4.3.2" in insp.possible_clause_ids
        assert "4.3.3" in insp.possible_clause_ids

    def test_possible_clause_ids_are_deduplicated(self, tmp_path: Path):
        content = "**4.3.1** Text.\n\n**4.3.1** Duplicate.\n"
        doc = _load(tmp_path, content)
        insp = inspect_markdown(doc)
        assert insp.possible_clause_ids.count("4.3.1") == 1

    def test_cross_references_found(self, tmp_path: Path):
        doc = _load(tmp_path, CLAUSE_DOC)
        insp = inspect_markdown(doc)
        assert "2.1.2" in insp.cross_reference_patterns
        assert "4.3.2" in insp.cross_reference_patterns
        assert "10.5" in insp.cross_reference_patterns

    def test_cross_references_deduplicated(self, tmp_path: Path):
        content = "See §4.3 and also §4.3 again.\n"
        doc = _load(tmp_path, content)
        insp = inspect_markdown(doc)
        assert insp.cross_reference_patterns.count("4.3") == 1


# ---------------------------------------------------------------------------
# 6. Inspection — immutability / no source mutation
# ---------------------------------------------------------------------------

class TestInspectionImmutability:

    def test_inspection_does_not_modify_raw_text(self, tmp_path: Path):
        content = "# Title\n\n**1.1.1** Clause text.\n"
        doc = _load(tmp_path, content)
        original_text = doc.raw_text
        inspect_markdown(doc)
        assert doc.raw_text == original_text

    def test_totals_mirror_document(self, tmp_path: Path):
        content = "# Title\n\nsome text\n"
        doc = _load(tmp_path, content)
        insp = inspect_markdown(doc)
        assert insp.total_lines == doc.line_count
        assert insp.total_characters == doc.character_count


# ---------------------------------------------------------------------------
# 7. Real corpus integration test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestRealCorpus:

    def test_corpus_loads_successfully(self):
        doc = load_policy_document(REAL_CORPUS)
        assert isinstance(doc, PolicyDocument)

    def test_corpus_raw_text_not_empty(self):
        doc = load_policy_document(REAL_CORPUS)
        assert len(doc.raw_text) > 0

    def test_corpus_raw_text_matches_file(self):
        doc = load_policy_document(REAL_CORPUS)
        expected = REAL_CORPUS.read_text(encoding="utf-8")
        assert doc.raw_text == expected

    def test_corpus_character_count(self):
        doc = load_policy_document(REAL_CORPUS)
        assert doc.character_count == len(doc.raw_text)

    def test_corpus_line_count(self):
        doc = load_policy_document(REAL_CORPUS)
        assert doc.line_count == len(doc.raw_text.splitlines())

    def test_corpus_has_headings(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert len(insp.headings) > 0

    def test_corpus_has_h1_and_h2_headings(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert 1 in insp.heading_counts_by_level
        assert 2 in insp.heading_counts_by_level

    def test_corpus_has_expected_part_headings(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        h1_texts = [h.text for h in insp.headings if h.level == 1]
        # Expect at least some Part headings
        part_headings = [t for t in h1_texts if t.startswith("Part")]
        assert len(part_headings) >= 10  # corpus has 12 parts

    def test_corpus_has_clause_ids(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert len(insp.possible_clause_ids) > 0

    def test_corpus_clause_id_4_3_2_present(self):
        """§4.3.2 is a key clause referenced throughout the manual."""
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert "4.3.2" in insp.possible_clause_ids

    def test_corpus_has_cross_references(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert len(insp.cross_reference_patterns) > 0

    def test_corpus_has_tables(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert insp.table_row_count > 0

    def test_corpus_has_lettered_list_items(self):
        doc = load_policy_document(REAL_CORPUS)
        insp = inspect_markdown(doc)
        assert insp.ordered_list_item_count > 0

    def test_corpus_inspection_does_not_mutate_raw_text(self):
        doc = load_policy_document(REAL_CORPUS)
        original = doc.raw_text
        inspect_markdown(doc)
        assert doc.raw_text == original
