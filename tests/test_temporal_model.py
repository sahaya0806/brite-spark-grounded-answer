"""
Tests for Day-2 Milestone 1 — Temporal Policy Data Model.

Verifies:
1. Existing PolicyClause construction still works with default temporal metadata.
2. Explicit effective_from and effective_to dates are stored and accessible.
3. Open-ended temporal validity (effective_to=None) is represented safely.
4. source_document is recorded and preserved.
5. Immutability and hashing behavior of frozen PolicyClause dataclass.
6. Existing 137-clause parser output populates default temporal metadata.
7. ClauseStore indexing and lookup remain compatible.
8. RetrievalResult wrapping of temporal PolicyClause instances works.
9. EvidenceEvaluator processes temporal PolicyClause instances without alteration.
10. Temporal metadata does not alter clause text, sub-items, or cross-references.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest

from src.evidence.evaluator import EvidenceEvaluator
from src.evidence.models import DecisionStatus
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import (
    ClauseSubItem,
    PolicyClause,
    parse_clauses,
)
from src.ingestion.store import ClauseStore
from src.retrieval.models import RetrievalResult

REAL_CORPUS = Path("data/raw/policy_manual.md")


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_sample_clause(
    clause_id: str = "4.3.2",
    effective_from: date | None = None,
    effective_to: date | None = None,
    source_document: str = "policy_manual.md",
) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        part_id="4",
        part_title="Part 4 — Exclusions",
        section_id="4.3",
        section_title="4.3 Recipient obligations",
        text="A recipient must report any change of circumstances within 10 calendar days.",
        sub_items=(),
        cross_references=("§8.5", "§10.5"),
        source_path=Path("data/raw/policy_manual.md"),
        start_line=200,
        end_line=205,
        effective_from=effective_from,
        effective_to=effective_to,
        source_document=source_document,
    )


# ---------------------------------------------------------------------------
# 1. Backward Compatibility & Defaults
# ---------------------------------------------------------------------------

class TestPolicyClauseDefaults:

    def test_construction_without_temporal_args(self):
        """Constructing PolicyClause without new kwargs uses default values."""
        clause = PolicyClause(
            clause_id="1.1.1",
            part_id="1",
            part_title="Part 1 — Scope and Definitions",
            section_id="1.1",
            section_title="1.1 Purpose of the Program",
            text="The Household Support Program provides monthly financial assistance.",
            sub_items=(),
            cross_references=(),
            source_path=Path("data/raw/policy_manual.md"),
            start_line=18,
            end_line=19,
        )
        assert clause.effective_from is None
        assert clause.effective_to is None
        assert clause.source_document == "policy_manual.md"

    def test_original_corpus_parsed_clauses_have_defaults(self):
        """Parsing real corpus assigns default temporal metadata to all 137 clauses."""
        if not REAL_CORPUS.exists():
            pytest.skip("Real corpus not found")

        doc = load_policy_document(REAL_CORPUS)
        clauses = parse_clauses(doc)
        assert len(clauses) == 137

        for c in clauses:
            assert c.effective_from is None
            assert c.effective_to is None
            assert c.source_document == "policy_manual.md"


# ---------------------------------------------------------------------------
# 2. Temporal Fields Representation
# ---------------------------------------------------------------------------

class TestTemporalMetadataRepresentation:

    def test_effective_from_date(self):
        """A clause can represent a specific effective_from date."""
        effective_start = date(2026, 3, 1)
        clause = _create_sample_clause(
            effective_from=effective_start,
            source_document="Amendment No. 2026-01.md",
        )
        assert clause.effective_from == effective_start
        assert clause.effective_from == date(2026, 3, 1)

    def test_effective_to_date(self):
        """A clause can represent a specific effective_to date (e.g. superseded)."""
        effective_end = date(2026, 2, 28)
        clause = _create_sample_clause(
            effective_to=effective_end,
            source_document="policy_manual.md",
        )
        assert clause.effective_to == effective_end
        assert clause.effective_to == date(2026, 2, 28)

    def test_bounded_effective_period(self):
        """A clause can represent both effective_from and effective_to."""
        start = date(2025, 1, 1)
        end = date(2026, 2, 28)
        clause = _create_sample_clause(effective_from=start, effective_to=end)
        assert clause.effective_from == start
        assert clause.effective_to == end

    def test_open_ended_effective_period(self):
        """Open-ended validity is represented safely via None."""
        clause = _create_sample_clause(
            effective_from=date(2026, 3, 1),
            effective_to=None,
        )
        assert clause.effective_from == date(2026, 3, 1)
        assert clause.effective_to is None

    def test_source_document_custom(self):
        """source_document accurately captures origin document identifier."""
        clause = _create_sample_clause(
            source_document="Amendment No. 2026-01.md",
        )
        assert clause.source_document == "Amendment No. 2026-01.md"


# ---------------------------------------------------------------------------
# 3. Immutability & Dataclass Behavior
# ---------------------------------------------------------------------------

class TestImmutabilityAndDataclassProperties:

    def test_frozen_immutability(self):
        """PolicyClause remains frozen and attributes cannot be modified."""
        clause = _create_sample_clause()
        with pytest.raises(Exception):  # FrozenInstanceError
            clause.effective_from = date(2026, 3, 1)  # type: ignore[misc]

    def test_equality_and_hashing(self):
        """Clauses with identical fields compare equal and have equal hashes."""
        c1 = _create_sample_clause(effective_from=date(2026, 3, 1))
        c2 = _create_sample_clause(effective_from=date(2026, 3, 1))
        c3 = _create_sample_clause(effective_from=date(2025, 1, 1))

        assert c1 == c2
        assert hash(c1) == hash(c2)
        assert c1 != c3
        assert hash(c1) != hash(c3)

    def test_temporal_metadata_preserves_clause_content_and_provenance(self):
        """Adding temporal metadata does not alter text, line tracking, or cross-references."""
        c = _create_sample_clause(
            effective_from=date(2026, 3, 1),
            effective_to=date(2026, 12, 31),
            source_document="Amendment No. 2026-01.md",
        )
        assert c.clause_id == "4.3.2"
        assert c.start_line == 200
        assert c.end_line == 205
        assert c.cross_references == ("§8.5", "§10.5")
        assert "10 calendar days" in c.text


# ---------------------------------------------------------------------------
# 4. Pipeline Component Compatibility
# ---------------------------------------------------------------------------

class TestPipelineCompatibility:

    def test_clause_store_compatibility(self):
        """ClauseStore successfully indexes and retrieves clauses with temporal fields."""
        c1 = _create_sample_clause(clause_id="4.3.2", effective_from=date(2026, 3, 1))
        store = ClauseStore([c1])

        retrieved = store.get_by_id("4.3.2")
        assert retrieved.effective_from == date(2026, 3, 1)
        assert retrieved.source_document == "policy_manual.md"

    def test_retrieval_result_compatibility(self):
        """RetrievalResult preserves temporal metadata on wrapped clause."""
        c = _create_sample_clause(effective_from=date(2026, 3, 1))
        rr = RetrievalResult(
            clause=c,
            semantic_score=0.8,
            lexical_score=0.7,
            combined_score=0.03,
            sources=("semantic", "lexical"),
        )
        assert rr.clause.effective_from == date(2026, 3, 1)

    def test_evidence_evaluator_compatibility(self):
        """EvidenceEvaluator evaluates RetrievalResults with temporal clauses without error."""
        c = _create_sample_clause(effective_from=date(2026, 3, 1))
        rr = RetrievalResult(
            clause=c,
            semantic_score=0.8,
            lexical_score=0.7,
            combined_score=0.03,
            sources=("semantic", "lexical"),
        )
        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("How many days to report a change?", [rr])
        assert decision.status in (
            DecisionStatus.SUPPORTED,
            DecisionStatus.INSUFFICIENT,
            DecisionStatus.CONFLICTING,
        )
        if decision.primary_clauses:
            assert decision.primary_clauses[0].effective_from == date(2026, 3, 1)
