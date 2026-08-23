"""
Tests for Day-2 Milestone 3 — Temporal Applicability / Policy Version Selection.

Verifies:
1. Determination-date rules (§6.4.1, §6.6.1, §10.5.2, §10.5.3A).
2. Change-of-circumstances rules (§4.3.2, §9.1.4).
3. Cross-trigger safety (providing wrong date type returns TEMPORAL_CONTEXT_REQUIRED).
4. Providing both dates resolves each clause according to its specific trigger.
5. Boundary dates (2026-02-28, 2026-03-01, 2026-03-02).
6. New clause §10.5.3A handling (NOT_YET_IN_FORCE before 2026-03-01, RESOLVED_AMENDMENT on/after).
7. Missing date safety (never silently default or guess).
8. Unamended clause handling (active across all dates without context requirement).
9. Immutability of original policy clauses and determinism of resolver.
10. Active corpus resolution across pre- and post-amendment dates.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest

from src.ingestion.amendment import (
    AmendmentDocument,
    TriggerType,
    parse_amendment,
)
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import PolicyClause, parse_clauses
from src.temporal import (
    ResolutionStatus,
    TemporalApplicabilityResolver,
    TemporalContext,
    TemporalResolution,
)

AMENDMENT_PATH = Path("data/raw/Amendment No. 2026-01.md")
ORIGINAL_MANUAL_PATH = Path("data/raw/policy_manual.md")


@pytest.fixture(scope="module")
def resolver() -> TemporalApplicabilityResolver:
    if not ORIGINAL_MANUAL_PATH.exists() or not AMENDMENT_PATH.exists():
        pytest.skip("Corpus files not found")

    orig_doc = load_policy_document(ORIGINAL_MANUAL_PATH)
    original_clauses = parse_clauses(orig_doc)

    amend_doc = load_policy_document(AMENDMENT_PATH)
    amendment = parse_amendment(amend_doc)

    return TemporalApplicabilityResolver(original_clauses, amendment)


# ---------------------------------------------------------------------------
# 1. Determination-Date Rules (§6.4.1(a), §6.6.1, §10.5.2)
# ---------------------------------------------------------------------------

class TestDeterminationDateRules:

    def test_earnings_disregard_pre_amendment(self, resolver):
        """Determination date 2026-02-15 selects original §6.4.1 ($120)."""
        ctx = TemporalContext(determination_date=date(2026, 2, 15))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert res.applicable_clause is not None
        assert "$120 per month" in res.applicable_clause.text
        assert res.source_document == "policy_manual.md"

    def test_earnings_disregard_effective_date(self, resolver):
        """Determination date 2026-03-01 selects amended §6.4.1 ($175)."""
        ctx = TemporalContext(determination_date=date(2026, 3, 1))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert res.applicable_clause is not None
        assert "$175 per month" in res.applicable_clause.text
        assert res.source_document == "Amendment No. 2026-01.md"

    def test_earnings_disregard_post_amendment(self, resolver):
        """Determination date 2026-04-15 selects amended §6.4.1 ($175)."""
        ctx = TemporalContext(determination_date=date(2026, 4, 15))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert res.applicable_clause is not None
        assert "$175 per month" in res.applicable_clause.text

    def test_income_thresholds_pre_amendment(self, resolver):
        """Determination date 2026-01-20 selects original §6.6.1 ($1,180 1-person)."""
        ctx = TemporalContext(determination_date=date(2026, 1, 20))
        res = resolver.resolve("6.6.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "$1,180" in res.applicable_clause.text

    def test_income_thresholds_post_amendment(self, resolver):
        """Determination date 2026-03-15 selects amended §6.6.1 ($1,225 1-person)."""
        ctx = TemporalContext(determination_date=date(2026, 3, 15))
        res = resolver.resolve("6.6.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "$1,225" in res.applicable_clause.text

    def test_sanctions_percentage_pre_amendment(self, resolver):
        """Determination date 2026-02-01 selects original §10.5.2 (20 per cent)."""
        ctx = TemporalContext(determination_date=date(2026, 2, 1))
        res = resolver.resolve("10.5.2", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "20 per cent" in res.applicable_clause.text

    def test_sanctions_percentage_post_amendment(self, resolver):
        """Determination date 2026-03-10 selects amended §10.5.2 (15 per cent)."""
        ctx = TemporalContext(determination_date=date(2026, 3, 10))
        res = resolver.resolve("10.5.2", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "15 per cent" in res.applicable_clause.text


# ---------------------------------------------------------------------------
# 2. Change-of-Circumstances Rules (§4.3.2, §9.1.4)
# ---------------------------------------------------------------------------

class TestChangeOfCircumstancesRules:

    def test_reporting_period_4_3_2_pre_amendment(self, resolver):
        """Change date 2026-02-15 selects original §4.3.2 (10 calendar days)."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 2, 15))
        res = resolver.resolve("4.3.2", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "10 calendar days" in res.applicable_clause.text
        assert res.source_document == "policy_manual.md"

    def test_reporting_period_4_3_2_effective_date(self, resolver):
        """Change date 2026-03-01 selects amended §4.3.2 (14 calendar days)."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 3, 1))
        res = resolver.resolve("4.3.2", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "14 calendar days" in res.applicable_clause.text
        assert res.source_document == "Amendment No. 2026-01.md"

    def test_reporting_period_4_3_2_post_amendment(self, resolver):
        """Change date 2026-04-15 selects amended §4.3.2 (14 calendar days)."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 4, 15))
        res = resolver.resolve("4.3.2", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "14 calendar days" in res.applicable_clause.text

    def test_overpayment_reporting_9_1_4_pre_amendment(self, resolver):
        """Change date 2026-02-10 selects original §9.1.4 (30 calendar days)."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 2, 10))
        res = resolver.resolve("9.1.4", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "30 calendar days" in res.applicable_clause.text

    def test_overpayment_reporting_9_1_4_post_amendment(self, resolver):
        """Change date 2026-03-10 selects amended §9.1.4 (14 calendar days)."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 3, 10))
        res = resolver.resolve("9.1.4", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "14 calendar days" in res.applicable_clause.text


# ---------------------------------------------------------------------------
# 3. Trigger Specificity & Cross-Trigger Safety
# ---------------------------------------------------------------------------

class TestTriggerSafety:

    def test_providing_only_determination_date_for_change_clause_requires_context(self, resolver):
        """Providing only determination_date for §4.3.2 must NOT resolve and must require change date."""
        ctx = TemporalContext(determination_date=date(2026, 4, 15))
        res = resolver.resolve("4.3.2", ctx)
        assert res.status == ResolutionStatus.TEMPORAL_CONTEXT_REQUIRED
        assert res.applicable_clause is None
        assert res.trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE
        assert "change_of_circumstances_date" in res.reason

    def test_providing_only_change_date_for_determination_clause_requires_context(self, resolver):
        """Providing only change_date for §6.4.1 must NOT resolve and must require determination date."""
        ctx = TemporalContext(change_of_circumstances_date=date(2026, 4, 15))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.TEMPORAL_CONTEXT_REQUIRED
        assert res.applicable_clause is None
        assert res.trigger_type == TriggerType.DETERMINATION_DATE
        assert "determination_date" in res.reason

    def test_providing_both_dates_resolves_each_clause_correctly(self, resolver):
        """
        Scenario:
        - Determination date: 2026-03-15 (POST-amendment)
        - Change of circumstances date: 2026-02-15 (PRE-amendment)

        Under Amendment §5.1 & §5.2:
        - Earnings disregard (§6.4.1) uses determination date -> $175 (AMENDMENT)
        - Change reporting (§4.3.2) uses change date -> 10 days (ORIGINAL)
        """
        ctx = TemporalContext(
            determination_date=date(2026, 3, 15),
            change_of_circumstances_date=date(2026, 2, 15),
        )

        res_earnings = resolver.resolve("6.4.1", ctx)
        res_reporting = resolver.resolve("4.3.2", ctx)

        assert res_earnings.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "$175 per month" in res_earnings.applicable_clause.text

        assert res_reporting.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "10 calendar days" in res_reporting.applicable_clause.text


# ---------------------------------------------------------------------------
# 4. Boundary Cases
# ---------------------------------------------------------------------------

class TestBoundaryDates:

    def test_day_before_amendment_is_pre(self, resolver):
        """2026-02-28 is strictly pre-amendment."""
        ctx = TemporalContext(determination_date=date(2026, 2, 28))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_ORIGINAL
        assert "$120 per month" in res.applicable_clause.text

    def test_effective_day_is_amendment(self, resolver):
        """2026-03-01 is amendment-effective."""
        ctx = TemporalContext(determination_date=date(2026, 3, 1))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "$175 per month" in res.applicable_clause.text

    def test_day_after_effective_day_is_amendment(self, resolver):
        """2026-03-02 is amendment-effective."""
        ctx = TemporalContext(determination_date=date(2026, 3, 2))
        res = resolver.resolve("6.4.1", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert "$175 per month" in res.applicable_clause.text


# ---------------------------------------------------------------------------
# 5. New Clause §10.5.3A
# ---------------------------------------------------------------------------

class TestNewClause1053A:

    def test_10_5_3A_pre_amendment_is_not_in_force(self, resolver):
        """Before 2026-03-01, new clause §10.5.3A was not yet in force."""
        ctx = TemporalContext(determination_date=date(2026, 2, 15))
        res = resolver.resolve("10.5.3A", ctx)
        assert res.status == ResolutionStatus.NOT_YET_IN_FORCE
        assert res.applicable_clause is None
        assert "not in force" in res.reason

    def test_10_5_3A_post_amendment_is_resolved(self, resolver):
        """On or after 2026-03-01, new clause §10.5.3A is active."""
        ctx = TemporalContext(determination_date=date(2026, 3, 15))
        res = resolver.resolve("10.5.3A", ctx)
        assert res.status == ResolutionStatus.RESOLVED_AMENDMENT
        assert res.applicable_clause is not None
        assert "increased the award" in res.applicable_clause.text
        assert res.applicable_clause.clause_id == "10.5.3A"


# ---------------------------------------------------------------------------
# 6. Unamended Clauses & General Safety
# ---------------------------------------------------------------------------

class TestUnamendedAndSafety:

    def test_unamended_clause_active_without_context(self, resolver):
        """Unamended clauses (e.g. §2.1.1) return UNAMENDED even with None context."""
        res = resolver.resolve("2.1.1", context=None)
        assert res.status == ResolutionStatus.UNAMENDED
        assert res.applicable_clause is not None
        assert res.applicable_clause.clause_id == "2.1.1"

    def test_missing_date_never_guesses_for_amended_clause(self, resolver):
        """No context provided for an amended clause returns TEMPORAL_CONTEXT_REQUIRED."""
        res = resolver.resolve("4.3.2", context=None)
        assert res.status == ResolutionStatus.TEMPORAL_CONTEXT_REQUIRED
        assert res.applicable_clause is None

    def test_unknown_clause_id_returns_not_found(self, resolver):
        """Unknown clause returns CLAUSE_NOT_FOUND."""
        res = resolver.resolve("99.99.99", TemporalContext(determination_date=date(2026, 3, 1)))
        assert res.status == ResolutionStatus.CLAUSE_NOT_FOUND
        assert res.applicable_clause is None

    def test_determinism(self, resolver):
        """Same input always produces identical resolution."""
        ctx = TemporalContext(determination_date=date(2026, 3, 1))
        res1 = resolver.resolve("6.4.1", ctx)
        res2 = resolver.resolve("6.4.1", ctx)
        assert res1 == res2

    def test_active_clauses_count_pre_and_post(self, resolver):
        """
        Pre-amendment: 137 active clauses (10.5.3A not yet in force).
        Post-amendment: 138 active clauses (137 base + 10.5.3A).
        """
        ctx_pre = TemporalContext(
            determination_date=date(2026, 2, 15),
            change_of_circumstances_date=date(2026, 2, 15),
        )
        active_pre = resolver.get_active_clauses(ctx_pre)
        assert len(active_pre) == 137

        ctx_post = TemporalContext(
            determination_date=date(2026, 3, 15),
            change_of_circumstances_date=date(2026, 3, 15),
        )
        active_post = resolver.get_active_clauses(ctx_post)
        assert len(active_post) == 138
        assert any(c.clause_id == "10.5.3A" for c in active_post)
