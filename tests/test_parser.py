"""
Tests for Milestone 3 — Clause-Level Parsing and Structured Clause Store.

Test groups
-----------
1.  Basic clause extraction
2.  Part and Section association
3.  Lettered sub-items
4.  Cross-reference extraction
5.  Source line tracking
6.  ClauseStore API
7.  Table handling
8.  Heading / sub-item non-inflation guards
9.  Determinism
10. Real corpus integration tests (137 clauses, key clause validation)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loader import load_policy_document, PolicyDocument
from src.ingestion.parser import (
    ClauseSubItem,
    PolicyClause,
    parse_clauses,
)
from src.ingestion.store import ClauseNotFoundError, ClauseStore


REAL_CORPUS = Path("data/raw/policy_manual.md")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(tmp_path: Path, content: str) -> PolicyDocument:
    p = tmp_path / "test_policy.md"
    p.write_text(content, encoding="utf-8")
    return load_policy_document(p)


def _parse(tmp_path: Path, content: str) -> list[PolicyClause]:
    return parse_clauses(_make_doc(tmp_path, content))


def _store(tmp_path: Path, content: str) -> ClauseStore:
    return ClauseStore(_parse(tmp_path, content))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_POLICY = """\
# Part 1 — Introduction

## 1.1 Purpose

**1.1.1** First clause text here.

**1.1.2** Second clause text here.
"""

MULTI_SECTION_POLICY = """\
# Part 2 — Eligibility

## 2.1 Basic conditions

**2.1.1** Eligibility condition one.

**2.1.2** Eligibility condition two.

## 2.2 Continuing eligibility

**2.2.1** Continuing eligibility rule.
"""

SUBITEM_POLICY = """\
# Part 4 — Exclusions

## 4.3 Recipient obligations

**4.3.1** A recipient must —

(a) provide information;

(b) attend interviews;

(c) report changes.

**4.3.2** Changes must be reported within 10 days.
"""

XREF_POLICY = """\
# Part 7 — Calculation

## 7.1 The award

**7.1.1** The award is calculated under §7.2, less income per §6.4.

**7.1.2** Where below $25, no award is made.

**7.1.3** See §5.4 for special cases, subject to §7.3.
"""

TABLE_POLICY = """\
# Part 6 — Income

## 6.6 Income thresholds

**6.6.1** Thresholds are —

| Household size | Monthly threshold |
|:--|:--|
| 1 | $1,000 |
| 2 | $1,500 |

**6.6.2** Another clause after the table.
"""

MULTI_PART_POLICY = """\
# Part 1 — Scope

## 1.1 Purpose

**1.1.1** Clause in Part 1.

# Part 2 — Eligibility

## 2.1 Basic conditions

**2.1.1** Clause in Part 2.
"""


# ---------------------------------------------------------------------------
# 1. Basic clause extraction
# ---------------------------------------------------------------------------

class TestBasicExtraction:

    def test_single_clause_extracted(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert len(clauses) == 2

    def test_clause_id_correct(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert clauses[0].clause_id == "1.1.1"
        assert clauses[1].clause_id == "1.1.2"

    def test_clause_text_preserved(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert "First clause text here" in clauses[0].text
        assert "Second clause text here" in clauses[1].text

    def test_clause_text_includes_bold_opener(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert clauses[0].text.startswith("**1.1.1**")

    def test_clauses_in_document_order(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_SECTION_POLICY)
        ids = [c.clause_id for c in clauses]
        assert ids == ["2.1.1", "2.1.2", "2.2.1"]

    def test_returns_list_of_policy_clause(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        for c in clauses:
            assert isinstance(c, PolicyClause)

    def test_empty_policy_returns_no_clauses(self, tmp_path):
        content = "# Part 1 — Scope\n\n## 1.1 Purpose\n\nNo clauses here.\n"
        clauses = _parse(tmp_path, content)
        assert clauses == []


# ---------------------------------------------------------------------------
# 2. Part and Section association
# ---------------------------------------------------------------------------

class TestPartSectionAssociation:

    def test_part_id_assigned(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.part_id == "1" for c in clauses)

    def test_part_title_assigned(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all("Part 1" in c.part_title for c in clauses)

    def test_section_id_assigned(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.section_id == "1.1" for c in clauses)

    def test_section_title_assigned(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert "Purpose" in clauses[0].section_title

    def test_section_changes_between_sections(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_SECTION_POLICY)
        section_ids = [c.section_id for c in clauses]
        assert "2.1" in section_ids
        assert "2.2" in section_ids

    def test_part_changes_between_parts(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_PART_POLICY)
        assert clauses[0].part_id == "1"
        assert clauses[1].part_id == "2"

    def test_clause_knows_its_section(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_SECTION_POLICY)
        c221 = next(c for c in clauses if c.clause_id == "2.2.1")
        assert c221.section_id == "2.2"

    def test_source_path_is_absolute(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.source_path.is_absolute() for c in clauses)


# ---------------------------------------------------------------------------
# 3. Lettered sub-items
# ---------------------------------------------------------------------------

class TestSubItems:

    def test_sub_items_extracted(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        c431 = next(c for c in clauses if c.clause_id == "4.3.1")
        assert len(c431.sub_items) == 3

    def test_sub_item_identifiers(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        c431 = next(c for c in clauses if c.clause_id == "4.3.1")
        identifiers = [s.identifier for s in c431.sub_items]
        assert identifiers == ["a", "b", "c"]

    def test_sub_item_text_preserved(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        c431 = next(c for c in clauses if c.clause_id == "4.3.1")
        assert any("provide information" in s.text for s in c431.sub_items)

    def test_sub_items_do_not_create_separate_clauses(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        # Should have 4.3.1 and 4.3.2, not extra clauses for (a)(b)(c)
        assert len(clauses) == 2

    def test_clause_without_sub_items_has_empty_tuple(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.sub_items == () for c in clauses)

    def test_sub_item_is_clause_sub_item_type(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        c431 = next(c for c in clauses if c.clause_id == "4.3.1")
        for item in c431.sub_items:
            assert isinstance(item, ClauseSubItem)

    def test_sibling_clause_has_no_cross_contamination(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        c432 = next(c for c in clauses if c.clause_id == "4.3.2")
        assert len(c432.sub_items) == 0


# ---------------------------------------------------------------------------
# 4. Cross-reference extraction
# ---------------------------------------------------------------------------

class TestCrossReferences:

    def test_cross_references_extracted(self, tmp_path):
        clauses = _parse(tmp_path, XREF_POLICY)
        c711 = next(c for c in clauses if c.clause_id == "7.1.1")
        assert "§7.2" in c711.cross_references
        assert "§6.4" in c711.cross_references

    def test_multiple_cross_references_in_one_clause(self, tmp_path):
        clauses = _parse(tmp_path, XREF_POLICY)
        c713 = next(c for c in clauses if c.clause_id == "7.1.3")
        assert "§5.4" in c713.cross_references
        assert "§7.3" in c713.cross_references

    def test_cross_references_deduplicated(self, tmp_path):
        content = (
            "# Part 1 — Scope\n\n"
            "## 1.1 Purpose\n\n"
            "**1.1.1** See §2.1 and also §2.1 again.\n"
        )
        clauses = _parse(tmp_path, content)
        refs = clauses[0].cross_references
        assert refs.count("§2.1") == 1

    def test_clause_without_xrefs_has_empty_tuple(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.cross_references == () for c in clauses)

    def test_cross_references_in_first_occurrence_order(self, tmp_path):
        clauses = _parse(tmp_path, XREF_POLICY)
        c713 = next(c for c in clauses if c.clause_id == "7.1.3")
        # §5.4 appears before §7.3 in the text
        idx5 = c713.cross_references.index("§5.4")
        idx7 = c713.cross_references.index("§7.3")
        assert idx5 < idx7

    def test_cross_ref_preserves_section_symbol(self, tmp_path):
        clauses = _parse(tmp_path, XREF_POLICY)
        c711 = next(c for c in clauses if c.clause_id == "7.1.1")
        for ref in c711.cross_references:
            assert ref.startswith("§")


# ---------------------------------------------------------------------------
# 5. Source line tracking
# ---------------------------------------------------------------------------

class TestSourceLineTracking:

    def test_start_line_is_positive(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.start_line > 0 for c in clauses)

    def test_end_line_gte_start_line(self, tmp_path):
        clauses = _parse(tmp_path, SIMPLE_POLICY)
        assert all(c.end_line >= c.start_line for c in clauses)

    def test_clauses_non_overlapping(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_SECTION_POLICY)
        for i in range(len(clauses) - 1):
            assert clauses[i].end_line < clauses[i + 1].start_line

    def test_start_line_points_to_opener(self, tmp_path):
        """The opener line contains the **N.N.N** pattern."""
        doc = _make_doc(tmp_path, SIMPLE_POLICY)
        clauses = parse_clauses(doc)
        lines = doc.raw_text.splitlines()
        for c in clauses:
            opener_line = lines[c.start_line - 1]
            assert f"**{c.clause_id}**" in opener_line


# ---------------------------------------------------------------------------
# 6. ClauseStore API
# ---------------------------------------------------------------------------

class TestClauseStore:

    def test_count_matches_clause_list(self, tmp_path):
        store = _store(tmp_path, MULTI_SECTION_POLICY)
        assert store.count() == 3

    def test_get_by_id_returns_correct_clause(self, tmp_path):
        store = _store(tmp_path, MULTI_SECTION_POLICY)
        c = store.get_by_id("2.2.1")
        assert c.clause_id == "2.2.1"

    def test_get_by_id_raises_on_missing(self, tmp_path):
        store = _store(tmp_path, SIMPLE_POLICY)
        with pytest.raises(ClauseNotFoundError):
            store.get_by_id("99.99.99")

    def test_all_returns_all_clauses(self, tmp_path):
        store = _store(tmp_path, MULTI_SECTION_POLICY)
        all_clauses = store.all()
        assert len(all_clauses) == 3

    def test_all_returns_copy(self, tmp_path):
        store = _store(tmp_path, SIMPLE_POLICY)
        result = store.all()
        result.clear()
        assert store.count() == 2  # original store unaffected

    def test_repr_contains_count(self, tmp_path):
        store = _store(tmp_path, SIMPLE_POLICY)
        assert "2" in repr(store)

    def test_all_in_document_order(self, tmp_path):
        store = _store(tmp_path, MULTI_SECTION_POLICY)
        ids = [c.clause_id for c in store.all()]
        assert ids == ["2.1.1", "2.1.2", "2.2.1"]


# ---------------------------------------------------------------------------
# 7. Table handling
# ---------------------------------------------------------------------------

class TestTableHandling:

    def test_table_rows_in_clause_text(self, tmp_path):
        clauses = _parse(tmp_path, TABLE_POLICY)
        c661 = next(c for c in clauses if c.clause_id == "6.6.1")
        assert "|" in c661.text

    def test_table_does_not_create_spurious_clauses(self, tmp_path):
        clauses = _parse(tmp_path, TABLE_POLICY)
        ids = {c.clause_id for c in clauses}
        assert ids == {"6.6.1", "6.6.2"}

    def test_clause_after_table_has_correct_text(self, tmp_path):
        clauses = _parse(tmp_path, TABLE_POLICY)
        c662 = next(c for c in clauses if c.clause_id == "6.6.2")
        assert "Another clause after the table" in c662.text

    def test_table_clause_has_correct_section(self, tmp_path):
        clauses = _parse(tmp_path, TABLE_POLICY)
        c661 = next(c for c in clauses if c.clause_id == "6.6.1")
        assert c661.section_id == "6.6"


# ---------------------------------------------------------------------------
# 8. Heading / sub-item non-inflation guards
# ---------------------------------------------------------------------------

class TestNonInflation:

    def test_h1_headings_not_clauses(self, tmp_path):
        content = (
            "# Part 1 — Scope\n\n"
            "## 1.1 Purpose\n\n"
            "**1.1.1** Real clause.\n"
        )
        clauses = _parse(tmp_path, content)
        ids = {c.clause_id for c in clauses}
        assert "1" not in ids      # Part 1 heading is not a clause
        assert "1.1" not in ids    # Section heading is not a clause
        assert "1.1.1" in ids

    def test_h2_headings_not_clauses(self, tmp_path):
        clauses = _parse(tmp_path, MULTI_SECTION_POLICY)
        ids = {c.clause_id for c in clauses}
        assert "2.1" not in ids
        assert "2.2" not in ids

    def test_lettered_items_not_clauses(self, tmp_path):
        clauses = _parse(tmp_path, SUBITEM_POLICY)
        ids = {c.clause_id for c in clauses}
        # Sub-items (a)(b)(c) must not produce clauses
        assert len(ids) == 2
        assert "4.3.1" in ids
        assert "4.3.2" in ids


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_output_on_repeated_calls(self, tmp_path):
        doc = _make_doc(tmp_path, MULTI_SECTION_POLICY)
        clauses_a = parse_clauses(doc)
        clauses_b = parse_clauses(doc)
        assert len(clauses_a) == len(clauses_b)
        for a, b in zip(clauses_a, clauses_b):
            assert a.clause_id == b.clause_id
            assert a.text == b.text


# ---------------------------------------------------------------------------
# 10. Real corpus integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestRealCorpus:

    @pytest.fixture(scope="class")
    def store(self):
        doc = load_policy_document(REAL_CORPUS)
        return ClauseStore(parse_clauses(doc))

    @pytest.fixture(scope="class")
    def doc(self):
        return load_policy_document(REAL_CORPUS)

    # -- Clause count -------------------------------------------------------

    def test_exactly_137_clauses(self, store):
        assert store.count() == 137

    # -- Part/Section associations ------------------------------------------

    def test_clause_1_1_1_in_part_1(self, store):
        c = store.get_by_id("1.1.1")
        assert c.part_id == "1"
        assert c.section_id == "1.1"

    def test_clause_4_3_2_in_part_4_section_4_3(self, store):
        c = store.get_by_id("4.3.2")
        assert c.part_id == "4"
        assert c.section_id == "4.3"

    def test_clause_9_1_4_in_part_9_section_9_1(self, store):
        c = store.get_by_id("9.1.4")
        assert c.part_id == "9"
        assert c.section_id == "9.1"

    def test_clause_12_3_3_in_part_12(self, store):
        c = store.get_by_id("12.3.3")
        assert c.part_id == "12"

    # -- Contradiction preservation (§4.3.2 vs §9.1.4) ---------------------

    def test_4_3_2_contains_10_calendar_days(self, store):
        c = store.get_by_id("4.3.2")
        assert "10 calendar days" in c.text

    def test_9_1_4_contains_30_calendar_days(self, store):
        c = store.get_by_id("9.1.4")
        assert "30 calendar days" in c.text

    def test_contradiction_not_reconciled(self, store):
        """Both the 10-day and 30-day texts must coexist unmodified."""
        c432 = store.get_by_id("4.3.2")
        c914 = store.get_by_id("9.1.4")
        assert "10 calendar days" in c432.text
        assert "30 calendar days" in c914.text
        # The 10-day clause must NOT say 30
        assert "30 calendar days" not in c432.text
        # The 30-day clause must NOT say 10
        assert "10 calendar days" not in c914.text

    # -- Gap / cross-reference preservation (§7.1.3 → §5.4) ---------------

    def test_7_1_3_has_xref_to_5_4(self, store):
        c = store.get_by_id("7.1.3")
        assert "§5.4" in c.cross_references

    def test_7_1_3_xref_not_resolved(self, store):
        """Parser must NOT have silently corrected or removed the §5.4 ref."""
        c = store.get_by_id("7.1.3")
        assert "§5.4" in c.text  # still in raw text
        assert "§5.4" in c.cross_references  # still in metadata

    # -- Sub-item handling --------------------------------------------------

    def test_2_1_2_has_six_sub_items(self, store):
        c = store.get_by_id("2.1.2")
        assert len(c.sub_items) == 6

    def test_4_3_1_has_four_sub_items(self, store):
        c = store.get_by_id("4.3.1")
        assert len(c.sub_items) == 4

    def test_6_4_1_has_seven_sub_items(self, store):
        c = store.get_by_id("6.4.1")
        assert len(c.sub_items) == 7

    def test_sub_item_identifiers_are_letters(self, store):
        c = store.get_by_id("2.1.2")
        for item in c.sub_items:
            assert item.identifier.isalpha()

    # -- Table handling -----------------------------------------------------

    def test_6_6_1_contains_table(self, store):
        c = store.get_by_id("6.6.1")
        assert "|" in c.text

    def test_7_2_1_contains_table(self, store):
        c = store.get_by_id("7.2.1")
        assert "|" in c.text

    def test_table_not_extra_clauses(self, store):
        """Tables must not inflate the clause count."""
        assert store.count() == 137

    # -- Cross-references ---------------------------------------------------

    def test_9_1_4_xref_to_4_3(self, store):
        c = store.get_by_id("9.1.4")
        assert "§4.3" in c.cross_references

    def test_cross_refs_have_section_symbol(self, store):
        for c in store.all():
            for ref in c.cross_references:
                assert ref.startswith("§"), f"Clause {c.clause_id}: bad ref {ref!r}"

    # -- Source text immutability -------------------------------------------

    def test_raw_text_not_modified(self, doc, store):
        original = REAL_CORPUS.read_text(encoding="utf-8")
        assert doc.raw_text == original

    def test_clause_text_verbatim_in_source(self, doc, store):
        """Each clause's text should appear somewhere in the raw source."""
        raw = doc.raw_text
        for c in store.all():
            # First line of the clause text must appear in the raw source
            first_line = c.text.splitlines()[0]
            assert first_line in raw, (
                f"Clause {c.clause_id} opener not found in raw source"
            )

    # -- Source line tracking -----------------------------------------------

    def test_start_lines_strictly_ascending(self, store):
        clauses = store.all()
        for i in range(len(clauses) - 1):
            assert clauses[i].start_line < clauses[i + 1].start_line

    def test_end_lines_gte_start_lines(self, store):
        for c in store.all():
            assert c.end_line >= c.start_line

    # -- Lookup API ---------------------------------------------------------

    def test_get_by_id_succeeds_for_all_clauses(self, store):
        for c in store.all():
            fetched = store.get_by_id(c.clause_id)
            assert fetched.clause_id == c.clause_id

    def test_get_by_id_missing_raises(self, store):
        with pytest.raises(ClauseNotFoundError):
            store.get_by_id("99.99.99")
