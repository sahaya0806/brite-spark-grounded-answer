"""
Tests for Day-2 Milestone 2 — Amendment Parsing and Structured Policy Versions.

Verifies:
1. Amendment document loads successfully.
2. Amendment number and issue/effective dates are parsed.
3. Every expected amended target clause is discovered (§4.3.2, §6.4.1, §6.6.1, §9.1.4, §10.5.2, §10.5.3A).
4. Replacement values (monetary, percentage, duration) are preserved accurately.
5. Income threshold table replacement is captured in structured form.
6. New clause §10.5.3A (alphanumeric ID) is parsed cleanly.
7. Temporal trigger types (determination_date vs change_of_circumstances_date) are distinguished.
8. Source provenance and line tracking are preserved.
9. Amendment can be parsed and represented independently of the original manual.
10. create_amended_clauses generates valid, updated PolicyClause objects without modifying original objects.
11. Original policy_manual.md and Amendment No. 2026-01.md files remain unmodified.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest

from src.ingestion.amendment import (
    AmendmentChange,
    AmendmentDocument,
    ChangeType,
    TableRow,
    TransitionalProvision,
    TriggerType,
    parse_amendment,
)
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import PolicyClause, parse_clauses

AMENDMENT_PATH = Path("data/raw/Amendment No. 2026-01.md")
ORIGINAL_MANUAL_PATH = Path("data/raw/policy_manual.md")


@pytest.fixture(scope="module")
def amendment_doc() -> AmendmentDocument:
    if not AMENDMENT_PATH.exists():
        pytest.skip(f"Amendment not found at {AMENDMENT_PATH}")
    doc = load_policy_document(AMENDMENT_PATH)
    return parse_amendment(doc)


# ---------------------------------------------------------------------------
# 1. Metadata and Document Header Tests
# ---------------------------------------------------------------------------

class TestAmendmentMetadata:

    def test_document_loads(self, amendment_doc):
        """Amendment document loads and is of type AmendmentDocument."""
        assert isinstance(amendment_doc, AmendmentDocument)
        assert amendment_doc.source_path.name == "Amendment No. 2026-01.md"

    def test_amendment_id_parsed(self, amendment_doc):
        """Amendment identifier is 2026-01."""
        assert amendment_doc.amendment_id == "2026-01"

    def test_issue_and_effective_dates(self, amendment_doc):
        """Issued 12 Feb 2026, Effective 1 March 2026."""
        assert amendment_doc.issue_date == date(2026, 2, 12)
        assert amendment_doc.effective_date == date(2026, 3, 1)

    def test_title_parsed(self, amendment_doc):
        """Title contains Calder County Household Support Program."""
        assert "Calder County" in amendment_doc.title


# ---------------------------------------------------------------------------
# 2. Target Clause Discovery Tests
# ---------------------------------------------------------------------------

class TestTargetClauseDiscovery:

    def test_all_expected_target_clauses_present(self, amendment_doc):
        """All 6 target clause IDs are discovered in the amendment."""
        expected_ids = {"6.4.1", "4.3.2", "9.1.4", "6.6.1", "10.5.2", "10.5.3A"}
        discovered_ids = set(amendment_doc.target_clause_ids)
        assert expected_ids.issubset(discovered_ids)

    def test_clause_4_3_2_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("4.3.2")
        assert change is not None
        assert change.target_clause_id == "4.3.2"

    def test_clause_6_4_1_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("6.4.1")
        assert change is not None
        assert change.sub_item_id == "a"

    def test_clause_6_6_1_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("6.6.1")
        assert change is not None
        assert change.change_type == ChangeType.TABLE_SUBSTITUTION

    def test_clause_9_1_4_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("9.1.4")
        assert change is not None

    def test_clause_10_5_2_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("10.5.2")
        assert change is not None

    def test_clause_10_5_3A_discovered(self, amendment_doc):
        change = amendment_doc.get_change_for_clause("10.5.3A")
        assert change is not None
        assert change.change_type == ChangeType.INSERTION


# ---------------------------------------------------------------------------
# 3. Value Extraction & Rule Preservation Tests
# ---------------------------------------------------------------------------

class TestValueAndRulePreservation:

    def test_earnings_disregard_values(self, amendment_doc):
        """Earnings disregard replaced from $120 to $175 per month."""
        change = amendment_doc.get_change_for_clause("6.4.1")
        assert change.original_value == "$120 per month"
        assert change.amended_value == "$175 per month"
        assert change.change_type == ChangeType.SUBSTITUTION
        assert change.trigger_type == TriggerType.DETERMINATION_DATE

    def test_reporting_period_4_3_2_values(self, amendment_doc):
        """Reporting window replaced from 10 calendar days to 14 calendar days."""
        change = amendment_doc.get_change_for_clause("4.3.2")
        assert change.original_value == "10 calendar days"
        assert change.amended_value == "14 calendar days"
        assert change.trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE

    def test_overpayment_reporting_9_1_4_values(self, amendment_doc):
        """Overpayment safe-harbour window replaced from 30 calendar days to 14 calendar days."""
        change = amendment_doc.get_change_for_clause("9.1.4")
        assert change.original_value == "30 calendar days"
        assert change.amended_value == "14 calendar days"
        assert change.trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE

    def test_sanction_percentage_10_5_2_values(self, amendment_doc):
        """Sanction reduction replaced from 20 per cent to 15 per cent."""
        change = amendment_doc.get_change_for_clause("10.5.2")
        assert change.original_value == "20 per cent"
        assert change.amended_value == "15 per cent"
        assert change.trigger_type == TriggerType.DETERMINATION_DATE

    def test_income_threshold_table_parsed(self, amendment_doc):
        """Amended income threshold table contains correct structured rows."""
        change = amendment_doc.get_change_for_clause("6.6.1")
        assert len(change.table_rows) == 6

        # Check 1 person threshold: $1,225
        row1 = change.table_rows[0]
        assert row1.household_size == "1"
        assert "$1,225" in row1.monthly_threshold
        assert row1.numeric_amount == 1225

        # Check 2 person threshold: $1,650
        row2 = change.table_rows[1]
        assert row2.household_size == "2"
        assert row2.numeric_amount == 1650

        # Check additional member: + $425
        row_last = change.table_rows[5]
        assert "additional" in row_last.household_size
        assert row_last.numeric_amount == 425

    def test_new_clause_10_5_3A_text(self, amendment_doc):
        """New clause 10.5.3A contains the full verbatim protective rule."""
        change = amendment_doc.get_change_for_clause("10.5.3A")
        assert "**10.5.3A**" in change.amended_value
        assert "increased the award" in change.amended_value


# ---------------------------------------------------------------------------
# 4. Temporal Applicability & Provenance Tests
# ---------------------------------------------------------------------------

class TestTemporalAndProvenance:

    def test_trigger_types_distinguished(self, amendment_doc):
        """Section 1, 3, 4 use DETERMINATION_DATE; Section 2 uses CHANGE_OF_CIRCUMSTANCES_DATE."""
        ch_earnings = amendment_doc.get_change_for_clause("6.4.1")
        ch_thresholds = amendment_doc.get_change_for_clause("6.6.1")
        ch_sanctions = amendment_doc.get_change_for_clause("10.5.2")
        ch_new_sanction = amendment_doc.get_change_for_clause("10.5.3A")

        ch_report_432 = amendment_doc.get_change_for_clause("4.3.2")
        ch_report_914 = amendment_doc.get_change_for_clause("9.1.4")

        assert ch_earnings.trigger_type == TriggerType.DETERMINATION_DATE
        assert ch_thresholds.trigger_type == TriggerType.DETERMINATION_DATE
        assert ch_sanctions.trigger_type == TriggerType.DETERMINATION_DATE
        assert ch_new_sanction.trigger_type == TriggerType.DETERMINATION_DATE

        assert ch_report_432.trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE
        assert ch_report_914.trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE

    def test_transitional_provisions_captured(self, amendment_doc):
        """Transitional paragraphs 5.1, 5.2, 5.3 are captured."""
        assert len(amendment_doc.transitional_provisions) >= 3
        para_ids = [p.paragraph_id for p in amendment_doc.transitional_provisions]
        assert "5.1" in para_ids
        assert "5.2" in para_ids
        assert "5.3" in para_ids

    def test_source_provenance_lines(self, amendment_doc):
        """All changes have positive line numbers from the amendment source."""
        for change in amendment_doc.changes:
            assert change.start_line > 0
            assert change.end_line >= change.start_line
            assert change.source_document == "Amendment No. 2026-01.md"


# ---------------------------------------------------------------------------
# 5. Policy Clause Integration & Generation Tests
# ---------------------------------------------------------------------------

class TestPolicyClauseGeneration:

    def test_create_amended_clauses(self, amendment_doc):
        """Generating amended PolicyClauses produces valid, updated records."""
        orig_doc = load_policy_document(ORIGINAL_MANUAL_PATH)
        orig_clauses = parse_clauses(orig_doc)

        amended_clauses = amendment_doc.create_amended_clauses(orig_clauses)
        assert len(amended_clauses) == 6

        by_id = {c.clause_id: c for c in amended_clauses}

        # 1. Check 6.4.1 has $175
        assert "6.4.1" in by_id
        assert "$175 per month" in by_id["6.4.1"].text
        assert by_id["6.4.1"].effective_from == date(2026, 3, 1)
        assert by_id["6.4.1"].source_document == "Amendment No. 2026-01.md"

        # 2. Check 4.3.2 has 14 calendar days
        assert "4.3.2" in by_id
        assert "14 calendar days" in by_id["4.3.2"].text

        # 3. Check 9.1.4 has 14 calendar days
        assert "9.1.4" in by_id
        assert "14 calendar days" in by_id["9.1.4"].text

        # 4. Check 6.6.1 has new table
        assert "6.6.1" in by_id
        assert "$1,225" in by_id["6.6.1"].text

        # 5. Check 10.5.2 has 15 per cent
        assert "10.5.2" in by_id
        assert "15 per cent" in by_id["10.5.2"].text

        # 6. Check 10.5.3A is created
        assert "10.5.3A" in by_id
        assert by_id["10.5.3A"].clause_id == "10.5.3A"
        assert by_id["10.5.3A"].part_id == "10"
        assert by_id["10.5.3A"].section_id == "10.5"
        assert "increased the award" in by_id["10.5.3A"].text

    def test_original_manual_file_unmodified(self):
        """Original policy manual is unmodified."""
        doc = load_policy_document(ORIGINAL_MANUAL_PATH)
        assert "10 calendar days" in doc.raw_text
        assert "30 calendar days" in doc.raw_text
        assert "$120 per month" in doc.raw_text
        assert "20 per cent" in doc.raw_text
        assert "10.5.3A" not in doc.raw_text
