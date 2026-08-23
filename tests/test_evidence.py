"""
Tests for Milestone 5 — Evidence Evaluation and Decision Layer.

All unit tests use synthetic clauses and fake RetrievalResults.
Real-corpus integration tests use the actual policy manual, the
hybrid retriever with FakeEmbeddingProvider, and the EvidenceEvaluator.

Test groups
-----------
1.  DecisionStatus enum
2.  EvidenceItem model
3.  ConflictDetail model
4.  EvidenceDecision model
5.  Numeric fact extraction
6.  Support signal extraction
7.  Gap signal extraction
8.  Relevance scoring
9.  Contradiction detection
10. EvidenceEvaluator — unit tests with synthetic evidence
11. EvidenceEvaluator — integration tests with real corpus
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evidence import (
    ConflictDetail,
    DecisionStatus,
    EvidenceConfig,
    EvidenceDecision,
    EvidenceEvaluator,
    EvidenceItem,
    NumericFact,
    compute_relevance_score,
    detect_conflicts,
    extract_gap_signals,
    extract_numeric_facts,
    extract_support_signals,
)
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import PolicyClause, parse_clauses
from src.ingestion.store import ClauseStore
from src.retrieval import FakeEmbeddingProvider, HybridRetriever
from src.retrieval.models import RetrievalResult

REAL_CORPUS = Path("data/raw/policy_manual.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clause(
    cid: str,
    text: str,
    section_id: str = "1.1",
    part_id: str = "1",
    xrefs: tuple[str, ...] = (),
) -> PolicyClause:
    return PolicyClause(
        clause_id=cid,
        part_id=part_id,
        part_title=f"Part {part_id}",
        section_id=section_id,
        section_title=f"{section_id} Section",
        text=text,
        sub_items=(),
        cross_references=xrefs,
        source_path=Path("/fake/policy.md"),
        start_line=1,
        end_line=1,
    )


def _result(
    clause: PolicyClause,
    semantic: float = 0.5,
    lexical: float = 0.5,
    combined: float = 0.02,
) -> RetrievalResult:
    return RetrievalResult(
        clause=clause,
        semantic_score=semantic,
        lexical_score=lexical,
        combined_score=combined,
        sources=("semantic", "lexical"),
    )


def _item(
    clause: PolicyClause,
    relevance: float = 0.6,
    support: tuple[str, ...] = (),
    gaps: tuple[str, ...] = (),
    unrefs: tuple[str, ...] = (),
    semantic: float = 0.5,
    lexical: float = 0.5,
    combined: float = 0.02,
) -> EvidenceItem:
    return EvidenceItem(
        result=_result(clause, semantic, lexical, combined),
        relevance_score=relevance,
        support_signals=support,
        gap_signals=gaps,
        unresolved_cross_refs=unrefs,
    )


# ---------------------------------------------------------------------------
# 1. DecisionStatus enum
# ---------------------------------------------------------------------------

class TestDecisionStatus:

    def test_values_exist(self):
        assert DecisionStatus.SUPPORTED
        assert DecisionStatus.INSUFFICIENT
        assert DecisionStatus.CONFLICTING

    def test_string_values(self):
        assert DecisionStatus.SUPPORTED.value == "SUPPORTED"
        assert DecisionStatus.INSUFFICIENT.value == "INSUFFICIENT"
        assert DecisionStatus.CONFLICTING.value == "CONFLICTING"

    def test_is_str_enum(self):
        assert isinstance(DecisionStatus.SUPPORTED, str)


# ---------------------------------------------------------------------------
# 2. EvidenceItem model
# ---------------------------------------------------------------------------

class TestEvidenceItemModel:

    def test_clause_accessor(self):
        c = _clause("1.1.1", "A recipient must report.")
        item = _item(c)
        assert item.clause is c

    def test_is_frozen(self):
        c = _clause("1.1.1", "Text.")
        item = _item(c)
        with pytest.raises((AttributeError, TypeError)):
            item.relevance_score = 0.9  # type: ignore[misc]

    def test_relevance_score_accessible(self):
        c = _clause("1.1.1", "Text.")
        item = _item(c, relevance=0.75)
        assert item.relevance_score == 0.75

    def test_empty_signals(self):
        c = _clause("1.1.1", "Text.")
        item = _item(c)
        assert item.support_signals == ()
        assert item.gap_signals == ()
        assert item.unresolved_cross_refs == ()


# ---------------------------------------------------------------------------
# 3. ConflictDetail model
# ---------------------------------------------------------------------------

class TestConflictDetailModel:

    def test_is_frozen(self):
        ca = _clause("1.1.1", "A")
        cb = _clause("2.1.1", "B", section_id="2.1")
        cd = ConflictDetail(
            clause_a=ca, clause_b=cb,
            conflict_type="competing_duration_days",
            value_a="10 days", value_b="30 days",
            explanation="conflict",
        )
        with pytest.raises((AttributeError, TypeError)):
            cd.value_a = "changed"  # type: ignore[misc]

    def test_clauses_accessible(self):
        ca = _clause("1.1.1", "A")
        cb = _clause("2.1.1", "B", section_id="2.1")
        cd = ConflictDetail(
            clause_a=ca, clause_b=cb,
            conflict_type="type",
            value_a="x", value_b="y",
            explanation="exp",
        )
        assert cd.clause_a.clause_id == "1.1.1"
        assert cd.clause_b.clause_id == "2.1.1"


# ---------------------------------------------------------------------------
# 4. EvidenceDecision model
# ---------------------------------------------------------------------------

class TestEvidenceDecisionModel:

    def test_is_frozen(self):
        d = EvidenceDecision(
            status=DecisionStatus.INSUFFICIENT,
            question="Q",
            evidence=(),
            rationale="R",
            support_score=0.0,
            primary_clauses=(),
            conflict_details=(),
            missing_information="",
            recommended_action="refuse",
        )
        with pytest.raises((AttributeError, TypeError)):
            d.status = DecisionStatus.SUPPORTED  # type: ignore[misc]

    def test_fields_accessible(self):
        d = EvidenceDecision(
            status=DecisionStatus.SUPPORTED,
            question="Q",
            evidence=(),
            rationale="R",
            support_score=0.8,
            primary_clauses=(),
            conflict_details=(),
            missing_information="",
            recommended_action="generate_answer",
        )
        assert d.status == DecisionStatus.SUPPORTED
        assert d.question == "Q"
        assert d.support_score == 0.8


# ---------------------------------------------------------------------------
# 5. Numeric fact extraction
# ---------------------------------------------------------------------------

class TestNumericFactExtraction:

    def test_duration_extracted(self):
        facts = extract_numeric_facts("report within 10 calendar days")
        assert any(f.kind == "duration_days" and f.value == "10" for f in facts)

    def test_multiple_durations(self):
        facts = extract_numeric_facts("within 10 days or 30 calendar days")
        values = {f.value for f in facts if f.kind == "duration_days"}
        assert "10" in values
        assert "30" in values

    def test_monetary_extracted(self):
        facts = extract_numeric_facts("resources not exceed $4,000")
        assert any(f.kind == "monetary" for f in facts)

    def test_percentage_extracted(self):
        facts = extract_numeric_facts("reduced by 20 per cent")
        assert any(f.kind == "percentage" and f.value == "20" for f in facts)

    def test_no_facts_in_plain_text(self):
        facts = extract_numeric_facts("the applicant must reside in the county")
        assert len(facts) == 0

    def test_returns_tuple(self):
        result = extract_numeric_facts("10 days")
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# 6. Support signal extraction
# ---------------------------------------------------------------------------

class TestSupportSignals:

    def test_obligation_language_detected(self):
        c = _clause("1.1.1", "A recipient must report changes.")
        tokens = frozenset(["recipient", "report", "changes"])
        signals = extract_support_signals(c, tokens)
        assert "contains_obligation_language" in signals

    def test_duration_value_detected(self):
        c = _clause("1.1.1", "Report within 10 calendar days.")
        tokens = frozenset(["report", "days"])
        signals = extract_support_signals(c, tokens)
        assert "contains_duration_value" in signals

    def test_lexical_overlap_detected(self):
        c = _clause("1.1.1", "household income resources eligibility")
        tokens = frozenset(["household", "income", "resources", "eligibility"])
        signals = extract_support_signals(c, tokens)
        assert any("lexical_overlap" in s for s in signals)

    def test_sub_items_signal(self):
        from src.ingestion.parser import ClauseSubItem
        c = PolicyClause(
            clause_id="1.1.1", part_id="1", part_title="Part 1",
            section_id="1.1", section_title="1.1 Section",
            text="A recipient must —",
            sub_items=(
                ClauseSubItem("a", "(a) provide information"),
                ClauseSubItem("b", "(b) attend interviews"),
            ),
            cross_references=(),
            source_path=Path("/fake/p.md"),
            start_line=1, end_line=3,
        )
        signals = extract_support_signals(c, frozenset(["provide"]))
        assert "has_2_sub_items" in signals

    def test_empty_clause_no_signals(self):
        c = _clause("1.1.1", "x y z")
        signals = extract_support_signals(c, frozenset(["qqq"]))
        assert len(signals) == 0 or all(
            "lexical_overlap" not in s for s in signals
        )


# ---------------------------------------------------------------------------
# 7. Gap signal extraction
# ---------------------------------------------------------------------------

class TestGapSignals:

    def test_unresolved_xref_detected(self):
        c = _clause("7.1.3", "See §5.4 for students.", xrefs=("§5.4",))
        retrieved_ids = frozenset({"7.1.3"})  # §5.4 NOT in evidence
        gaps = extract_gap_signals(c, retrieved_ids, frozenset())
        assert any("unresolved_cross_refs" in g for g in gaps)

    def test_resolved_xref_no_gap(self):
        c = _clause("7.1.3", "See §5.4 for students.", xrefs=("§5.4",))
        retrieved_ids = frozenset({"7.1.3", "5.4"})  # §5.4 IS in evidence
        gaps = extract_gap_signals(c, retrieved_ids, frozenset())
        assert not any("unresolved_cross_refs" in g for g in gaps)

    def test_delegation_signal(self):
        c = _clause("7.1.3", "See §5.4 for details.", xrefs=("§5.4",))
        retrieved_ids = frozenset({"7.1.3"})
        gaps = extract_gap_signals(c, retrieved_ids, frozenset())
        assert "delegates_to_cross_reference" in gaps

    def test_no_gaps_when_no_xrefs(self):
        c = _clause("2.4.1", "Resources must not exceed $4,000.")
        retrieved_ids = frozenset({"2.4.1"})
        gaps = extract_gap_signals(c, retrieved_ids, frozenset())
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# 8. Relevance scoring
# ---------------------------------------------------------------------------

class TestRelevanceScoring:

    def test_score_in_0_1(self):
        c = _clause("1.1.1", "text")
        r = _result(c, semantic=0.8, lexical=0.7)
        score = compute_relevance_score(r, ("contains_obligation_language",))
        assert 0.0 <= score <= 1.0

    def test_higher_scores_with_signals(self):
        c = _clause("1.1.1", "text")
        r = _result(c, semantic=0.5, lexical=0.5)
        score_no_signals = compute_relevance_score(r, ())
        score_with_signals = compute_relevance_score(
            r, ("contains_obligation_language", "contains_duration_value")
        )
        assert score_with_signals >= score_no_signals

    def test_zero_scores_produce_low_relevance(self):
        c = _clause("1.1.1", "text")
        r = _result(c, semantic=0.0, lexical=0.0)
        score = compute_relevance_score(r, ())
        assert score == 0.0

    def test_semantic_only_lower_than_both(self):
        c = _clause("1.1.1", "text")
        r_both = _result(c, semantic=0.7, lexical=0.7)
        r_sem = _result(c, semantic=0.7, lexical=0.0)
        s_both = compute_relevance_score(r_both, ())
        s_sem = compute_relevance_score(r_sem, ())
        assert s_both > s_sem


# ---------------------------------------------------------------------------
# 9. Contradiction detection
# ---------------------------------------------------------------------------

class TestContradictionDetection:

    def _make_items(self, clauses_and_scores):
        items = []
        for c, rel in clauses_and_scores:
            items.append(_item(c, relevance=rel))
        return items

    def test_no_conflict_single_item(self):
        c = _clause("1.1.1", "Report within 10 days.")
        conflicts = detect_conflicts([_item(c, relevance=0.6)])
        assert conflicts == []

    def test_no_conflict_same_section(self):
        # Same section_id — should not trigger conflict
        c1 = _clause("1.1.1", "Report within 10 days. Recipient must comply.")
        c2 = _clause("1.1.2", "Report within 30 days. Recipient must notify.")
        # Both in section "1.1"
        conflicts = detect_conflicts([
            _item(c1, relevance=0.7),
            _item(c2, relevance=0.7),
        ])
        assert conflicts == []

    def test_conflict_different_sections_same_topic(self):
        # Different sections, same obligation topic, different day counts
        c1 = _clause(
            "4.3.2",
            "A recipient must report any change within 10 calendar days "
            "of the change occurring.",
            section_id="4.3",
        )
        c2 = _clause(
            "9.1.4",
            "Where the recipient reported the change within 30 calendar days "
            "required under section 4.3, no overpayment shall be established.",
            section_id="9.1",
        )
        conflicts = detect_conflicts([
            _item(c1, relevance=0.7),
            _item(c2, relevance=0.7),
        ])
        assert len(conflicts) > 0
        ids = {(c.clause_a.clause_id, c.clause_b.clause_id) for c in conflicts}
        assert ("4.3.2", "9.1.4") in ids

    def test_conflict_carries_values(self):
        c1 = _clause("4.3.2", "must report within 10 calendar days.", section_id="4.3")
        c2 = _clause("9.1.4", "within 30 calendar days required.", section_id="9.1")
        # Add shared vocabulary
        c1 = _clause("4.3.2",
                     "recipient must report any change in circumstances "
                     "within 10 calendar days of becoming aware.",
                     section_id="4.3")
        c2 = _clause("9.1.4",
                     "recipient reported the change in circumstances "
                     "within 30 calendar days as required.",
                     section_id="9.1")
        conflicts = detect_conflicts([
            _item(c1, relevance=0.7),
            _item(c2, relevance=0.7),
        ])
        if conflicts:
            c = conflicts[0]
            assert "10" in c.value_a or "10" in c.value_b
            assert "30" in c.value_a or "30" in c.value_b

    def test_no_conflict_low_relevance_excluded(self):
        c1 = _clause("1.1.1", "report within 10 calendar days recipient must.",
                     section_id="1.1")
        c2 = _clause("2.1.1", "report within 30 calendar days recipient must.",
                     section_id="2.1")
        # Both below conflict_min_relevance=0.10
        conflicts = detect_conflicts([
            _item(c1, relevance=0.05),
            _item(c2, relevance=0.05),
        ])
        assert conflicts == []

    def test_returns_list(self):
        c = _clause("1.1.1", "text")
        result = detect_conflicts([_item(c)])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 10. EvidenceEvaluator — unit tests with synthetic evidence
# ---------------------------------------------------------------------------

class TestEvidenceEvaluatorUnit:

    def setup_method(self):
        self.evaluator = EvidenceEvaluator()

    # -- Decision model structure --

    def test_returns_evidence_decision(self):
        c = _clause("1.1.1", "text")
        d = self.evaluator.evaluate("question", [_result(c)])
        assert isinstance(d, EvidenceDecision)

    def test_question_preserved(self):
        c = _clause("1.1.1", "text")
        d = self.evaluator.evaluate("my test question", [_result(c)])
        assert d.question == "my test question"

    def test_status_is_decision_status(self):
        c = _clause("1.1.1", "text")
        d = self.evaluator.evaluate("q", [_result(c)])
        assert isinstance(d.status, DecisionStatus)

    # -- Empty evidence --

    def test_empty_evidence_is_insufficient(self):
        d = self.evaluator.evaluate("anything", [])
        assert d.status == DecisionStatus.INSUFFICIENT

    def test_empty_evidence_recommended_refuse(self):
        d = self.evaluator.evaluate("anything", [])
        assert d.recommended_action == "refuse"

    def test_empty_evidence_zero_support_score(self):
        d = self.evaluator.evaluate("anything", [])
        assert d.support_score == 0.0

    # -- Clearly insufficient ---

    def test_weak_evidence_is_insufficient(self):
        # Clause with zero retrieval scores → very low relevance → insufficient
        c = _clause("1.1.1", "unrelated tangential text.")
        r = _result(c, semantic=0.0, lexical=0.0, combined=0.001)
        d = self.evaluator.evaluate("specific policy question", [r])
        assert d.status == DecisionStatus.INSUFFICIENT

    def test_insufficient_recommended_refuse(self):
        c = _clause("1.1.1", "unrelated text.")
        d = self.evaluator.evaluate("q", [_result(c, semantic=0.0, lexical=0.0)])
        assert d.recommended_action == "refuse"

    # -- Clearly supported --

    def test_strong_evidence_is_supported(self):
        c = _clause(
            "2.4.1",
            "A household is not eligible where the total countable resources "
            "of the household exceed $4,000.",
        )
        r = _result(c, semantic=0.85, lexical=0.90, combined=0.03)
        d = self.evaluator.evaluate(
            "What is the resource limit for a household?", [r]
        )
        assert d.status == DecisionStatus.SUPPORTED

    def test_supported_recommended_generate(self):
        c = _clause(
            "2.4.1",
            "A household is not eligible where resources exceed $4,000 "
            "and must meet eligibility conditions.",
        )
        r = _result(c, semantic=0.85, lexical=0.90, combined=0.03)
        d = self.evaluator.evaluate("resource limit household", [r])
        assert d.recommended_action == "generate_answer"

    def test_supported_has_primary_clauses(self):
        c = _clause(
            "2.4.1",
            "A household is not eligible where resources exceed $4,000 "
            "and must satisfy eligibility conditions.",
        )
        r = _result(c, semantic=0.85, lexical=0.90, combined=0.03)
        d = self.evaluator.evaluate("resource limit household", [r])
        if d.status == DecisionStatus.SUPPORTED:
            assert len(d.primary_clauses) > 0

    # -- Conflicting evidence --

    def test_conflict_detected_in_decision(self):
        c1 = _clause(
            "4.3.2",
            "recipient must report any change in circumstances "
            "within 10 calendar days of becoming aware.",
            section_id="4.3",
        )
        c2 = _clause(
            "9.1.4",
            "recipient reported the change in circumstances "
            "within 30 calendar days as required.",
            section_id="9.1",
        )
        results = [
            _result(c1, semantic=0.7, lexical=0.9, combined=0.025),
            _result(c2, semantic=0.6, lexical=0.8, combined=0.020),
        ]
        d = self.evaluator.evaluate(
            "How many days to report a change?", results
        )
        assert d.status == DecisionStatus.CONFLICTING

    def test_conflict_recommended_surface(self):
        c1 = _clause("4.3.2",
                     "recipient must report change within 10 calendar days.",
                     section_id="4.3")
        c2 = _clause("9.1.4",
                     "recipient reported change within 30 calendar days.",
                     section_id="9.1")
        d = self.evaluator.evaluate("report deadline", [
            _result(c1, semantic=0.7, lexical=0.9, combined=0.025),
            _result(c2, semantic=0.6, lexical=0.8, combined=0.020),
        ])
        if d.status == DecisionStatus.CONFLICTING:
            assert d.recommended_action == "surface_conflict"

    def test_conflict_has_conflict_details(self):
        c1 = _clause("4.3.2",
                     "recipient must report change within 10 calendar days.",
                     section_id="4.3")
        c2 = _clause("9.1.4",
                     "recipient reported change within 30 calendar days.",
                     section_id="9.1")
        d = self.evaluator.evaluate("report deadline", [
            _result(c1, semantic=0.7, lexical=0.9, combined=0.025),
            _result(c2, semantic=0.6, lexical=0.8, combined=0.020),
        ])
        if d.status == DecisionStatus.CONFLICTING:
            assert len(d.conflict_details) > 0

    # -- Cross-reference delegation as gap --

    def test_delegating_clause_insufficient(self):
        c = _clause(
            "7.1.3",
            "The needs figure for full-time students (see §5.4) is subject "
            "to adjustments.",
            xrefs=("§5.4",),
        )
        r = _result(c, semantic=0.8, lexical=0.8, combined=0.025)
        d = self.evaluator.evaluate(
            "What is the policy for full-time students?", [r]
        )
        # Top item delegates to unresolved ref AND it's the top item by retrieval
        assert d.status == DecisionStatus.INSUFFICIENT

    # -- Determinism --

    def test_same_result_on_repeated_calls(self):
        c = _clause("1.1.1", "A recipient must report changes within 10 days.")
        r = _result(c, semantic=0.7, lexical=0.8)
        d1 = self.evaluator.evaluate("report deadline", [r])
        d2 = self.evaluator.evaluate("report deadline", [r])
        assert d1.status == d2.status
        assert d1.support_score == d2.support_score

    # -- Evidence preserved --

    def test_evidence_items_populated(self):
        c = _clause("1.1.1", "text")
        r = _result(c)
        d = self.evaluator.evaluate("q", [r])
        assert len(d.evidence) == 1

    def test_evidence_items_sorted_by_relevance(self):
        c1 = _clause("1.1.1", "strong relevant text must report change", section_id="1.1")
        c2 = _clause("2.1.1", "weak text xyz", section_id="2.1", part_id="2")
        results = [
            _result(c2, semantic=0.1, lexical=0.1),
            _result(c1, semantic=0.9, lexical=0.9),
        ]
        d = self.evaluator.evaluate("q", results)
        scores = [e.relevance_score for e in d.evidence]
        assert scores == sorted(scores, reverse=True)

    # -- Conservative default --

    def test_ambiguous_evidence_defaults_insufficient(self):
        # Moderate scores, no strong signals → INSUFFICIENT
        c = _clause("1.1.1", "the policy mentions various rules.")
        r = _result(c, semantic=0.2, lexical=0.2)
        d = self.evaluator.evaluate("specific complex question", [r])
        assert d.status == DecisionStatus.INSUFFICIENT

    def test_numeric_question_matched(self):
        c = _clause("6.6.1",
                    "A household is not eligible where countable income exceeds "
                    "the applicable threshold. The thresholds are $1,180 for "
                    "one person and $1,590 for two persons.",
                    section_id="6.6")
        r = _result(c, semantic=0.8, lexical=0.9, combined=0.03)
        d = self.evaluator.evaluate("income threshold household", [r])
        assert d.status == DecisionStatus.SUPPORTED

    def test_configurable_threshold(self):
        # A high threshold makes evidence that is normally supported become INSUFFICIENT
        strict_cfg = EvidenceConfig(min_support=0.95, strong_item_score=0.95)
        evaluator = EvidenceEvaluator(strict_cfg)
        c = _clause("2.4.1",
                    "resources must not exceed $4,000 household eligibility "
                    "condition must satisfy requirements.",
                    section_id="2.4")
        r = _result(c, semantic=0.7, lexical=0.7, combined=0.03)
        # Default evaluator supports this
        default_decision = self.evaluator.evaluate("resource limit", [r])
        assert default_decision.status == DecisionStatus.SUPPORTED
        # Strict evaluator marks INSUFFICIENT
        d = evaluator.evaluate("resource limit", [r])
        assert d.status == DecisionStatus.INSUFFICIENT

    def test_supporting_clause_ids_property(self):
        c = _clause("2.4.1", "resources must not exceed $4,000 household eligibility.")
        r = _result(c, semantic=0.85, lexical=0.90, combined=0.03)
        d = self.evaluator.evaluate("resource limit household", [r])
        assert d.status == DecisionStatus.SUPPORTED
        assert d.supporting_clause_ids == ("2.4.1",)

    def test_multiple_supporting_clauses(self):
        c1 = _clause("2.1.2", "The conditions are countable resources and income.", section_id="2.1")
        c2 = _clause("2.4.1", "Resources must not exceed $4,000 household eligibility.", section_id="2.4")
        results = [
            _result(c1, semantic=0.8, lexical=0.85, combined=0.03),
            _result(c2, semantic=0.85, lexical=0.90, combined=0.035),
        ]
        d = self.evaluator.evaluate("resource limit household conditions", results)
        assert d.status == DecisionStatus.SUPPORTED
        assert len(d.primary_clauses) >= 1
        for cid in d.supporting_clause_ids:
            assert cid in {"2.1.2", "2.4.1"}

    def test_duplicate_retrieval_results_deduplicated(self):
        c = _clause("2.4.1", "resources must not exceed $4,000 household eligibility.")
        r = _result(c, semantic=0.85, lexical=0.90, combined=0.03)
        # Pass identical result twice
        d = self.evaluator.evaluate("resource limit household", [r, r])
        assert len(d.evidence) == 1
        assert d.status == DecisionStatus.SUPPORTED

    def test_different_scopes_no_false_conflict(self):
        # Different subjects: one about absence duration, another about review period
        c1 = _clause(
            "3.2.1",
            "A temporary absence from the county does not exceed 28 days.",
            section_id="3.2",
        )
        c2 = _clause(
            "11.1.2",
            "An application for formal review must be submitted within 60 days.",
            section_id="11.1",
        )
        results = [
            _result(c1, semantic=0.5, lexical=0.5),
            _result(c2, semantic=0.5, lexical=0.5),
        ]
        conflicts = detect_conflicts([_item(c1, relevance=0.5), _item(c2, relevance=0.5)])
        assert conflicts == []

    def test_clause_ids_strictly_from_retrieved_evidence(self):
        c1 = _clause("1.1.1", "household eligibility rules")
        r = _result(c1, semantic=0.8, lexical=0.8)
        d = self.evaluator.evaluate("eligibility", [r])
        # No hallucinated clause IDs
        for c in d.primary_clauses:
            assert c.clause_id == "1.1.1"

    def test_missing_information_on_insufficient(self):
        d = self.evaluator.evaluate("anything", [])
        assert len(d.missing_information) > 0


# ---------------------------------------------------------------------------
# 11. EvidenceEvaluator — integration with real corpus
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestRealCorpusEvidence:

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Returns (retriever, evaluator) built from the real corpus."""
        doc = load_policy_document(REAL_CORPUS)
        store = ClauseStore(parse_clauses(doc))
        retriever = HybridRetriever.build(
            store.all(), FakeEmbeddingProvider(dim=64)
        )
        evaluator = EvidenceEvaluator()
        return retriever, evaluator

    # -- Case A: Known contradiction (§4.3.2 vs §9.1.4) -------------------

    def test_case_a_status_conflicting(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        assert decision.status == DecisionStatus.CONFLICTING, (
            f"Expected CONFLICTING, got {decision.status}. "
            f"Rationale: {decision.rationale}"
        )

    def test_case_a_conflict_involves_4_3_2(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        conflict_ids = set()
        for cd in decision.conflict_details:
            conflict_ids.add(cd.clause_a.clause_id)
            conflict_ids.add(cd.clause_b.clause_id)
        assert "4.3.2" in conflict_ids, (
            f"§4.3.2 not in conflict detail clause IDs: {conflict_ids}"
        )

    def test_case_a_conflict_involves_9_1_4(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        conflict_ids = set()
        for cd in decision.conflict_details:
            conflict_ids.add(cd.clause_a.clause_id)
            conflict_ids.add(cd.clause_b.clause_id)
        assert "9.1.4" in conflict_ids, (
            f"§9.1.4 not in conflict detail clause IDs: {conflict_ids}"
        )

    def test_case_a_recommended_surface_conflict(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        assert decision.recommended_action == "surface_conflict"

    def test_case_a_conflict_preserves_10_and_30_days(self, pipeline):
        """The contradiction values must be preserved, not reconciled."""
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        all_values = " ".join(
            f"{cd.value_a} {cd.value_b}" for cd in decision.conflict_details
        )
        assert "10" in all_values, "10-day value missing from conflict details"
        assert "30" in all_values, "30-day value missing from conflict details"

    def test_case_a_no_definitive_answer_generated(self, pipeline):
        """CONFLICTING must not trigger answer generation."""
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        assert decision.recommended_action != "generate_answer"

    # -- Case B: Apparent gap (§7.1.3 → §5.4 / full-time students) --------

    def test_case_b_status_insufficient(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the policy for full-time students?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        assert decision.status == DecisionStatus.INSUFFICIENT, (
            f"Expected INSUFFICIENT, got {decision.status}. "
            f"Rationale: {decision.rationale}"
        )

    def test_case_b_recommended_refuse(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the policy for full-time students?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        assert decision.recommended_action == "refuse"

    def test_case_b_does_not_resolve_gap(self, pipeline):
        """
        §7.1.3 references §5.4 for full-time students, but §5.4 is about
        care allowances.  The evaluator must not claim this is answered.
        """
        retriever, evaluator = pipeline
        q = "What is the policy for full-time students?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        assert decision.status != DecisionStatus.SUPPORTED, (
            "Gap case must not be marked SUPPORTED"
        )

    # -- Case C: Clearly covered policy -----------------------------------

    def test_case_c_status_supported(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the resource limit for a household?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        assert decision.status == DecisionStatus.SUPPORTED, (
            f"Expected SUPPORTED, got {decision.status}. "
            f"Rationale: {decision.rationale}"
        )

    def test_case_c_recommended_generate_answer(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the resource limit for a household?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        if decision.status == DecisionStatus.SUPPORTED:
            assert decision.recommended_action == "generate_answer"

    def test_case_c_has_primary_clauses(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the resource limit for a household?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        if decision.status == DecisionStatus.SUPPORTED:
            assert len(decision.primary_clauses) > 0

    # -- General properties ---

    def test_evidence_items_have_clause_ids(self, pipeline):
        retriever, evaluator = pipeline
        q = "What is the resource limit for a household?"
        results = retriever.retrieve(q, top_k=10)
        decision = evaluator.evaluate(q, results)
        for item in decision.evidence:
            assert item.clause.clause_id

    def test_evidence_items_have_source_lines(self, pipeline):
        retriever, evaluator = pipeline
        q = "resource limit household eligibility"
        results = retriever.retrieve(q, top_k=5)
        decision = evaluator.evaluate(q, results)
        for item in decision.evidence:
            assert item.clause.start_line > 0

    def test_deterministic_on_real_corpus(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many days to report a change?"
        results = retriever.retrieve(q, top_k=15)
        d1 = evaluator.evaluate(q, results)
        d2 = evaluator.evaluate(q, results)
        assert d1.status == d2.status
        assert d1.support_score == d2.support_score

    def test_empty_query_insufficient(self, pipeline):
        retriever, evaluator = pipeline
        results = retriever.retrieve("", top_k=5)
        decision = evaluator.evaluate("", results)
        assert decision.status == DecisionStatus.INSUFFICIENT

    def test_conflict_details_have_explanation(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        for cd in decision.conflict_details:
            assert len(cd.explanation) > 0

    def test_conflict_clauses_have_source_paths(self, pipeline):
        retriever, evaluator = pipeline
        q = "How many calendar days does a recipient have to report a change?"
        results = retriever.retrieve(q, top_k=15)
        decision = evaluator.evaluate(q, results)
        for cd in decision.conflict_details:
            assert cd.clause_a.source_path is not None
            assert cd.clause_b.source_path is not None
