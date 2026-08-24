"""
Tests for Day-2 Milestone 4 — Temporal Filter + Pipeline + CLI Integration.

Verifies:
1. CLI --date argument parsing and invalid date validation.
2. End-to-end date-aware question answering on change reporting period (§4.3.2 / §9.1.4).
3. Pipeline temporal filtering on earnings disregard (§6.4.1(a)).
4. Date boundary behavior (2026-02-28, 2026-03-01, 2026-03-02).
5. Safe refusal for date-sensitive questions asked without a date.
6. Non-temporal / unamended queries continue to work without a date.
7. Verification that temporal versions are not treated as simultaneous conflicts.
8. Citation formatting preserving exact provenance and amendment tags.
9. New clause §10.5.3A end-to-end behavior pre- and post-amendment.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest
from typer.testing import CliRunner

from src.app import app
from src.citation.renderer import format_clause_citation
from src.evidence.models import DecisionStatus
from src.generation.models import GroundedAnswer
from src.generation.providers import FakeChatProvider
from src.pipeline import PolicyQAPipeline
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.models import RetrievalResult
from src.temporal import TemporalContext

REAL_CORPUS = Path("data/raw/policy_manual.md")
REAL_AMENDMENT = Path("data/raw/Amendment No. 2026-01.md")


@pytest.fixture(scope="module")
def temporal_pipeline() -> PolicyQAPipeline:
    if not REAL_CORPUS.exists() or not REAL_AMENDMENT.exists():
        pytest.skip("Corpus files not present")

    return PolicyQAPipeline.build_from_corpus(
        corpus_path=REAL_CORPUS,
        amendment_path=REAL_AMENDMENT,
        embedding_provider=FakeEmbeddingProvider(dim=64),
        chat_provider=FakeChatProvider(),
    )


# ---------------------------------------------------------------------------
# 1. CLI Tests
# ---------------------------------------------------------------------------

class TestTemporalCLI:

    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_invalid_date_format_rejected(self):
        """Passing an invalid date produces a clear error with exit code 1."""
        result = self.runner.invoke(app, ["ask", "What is the earnings disregard?", "--date", "invalid-date"])
        assert result.exit_code != 0
        assert "Error: Invalid date format" in result.output
        assert "Expected YYYY-MM-DD" in result.output

    def test_cli_ask_with_date_reaches_pipeline(self, monkeypatch):
        """--date is parsed and passed to the pipeline."""
        captured_dates = []

        class MockPipeline:
            def ask(self, question, context=None, date=None, top_k=10):
                captured_dates.append(date)
                return GroundedAnswer(
                    question=question,
                    answer_text="Mock answer",
                    status=DecisionStatus.SUPPORTED,
                    citations=(),
                    supporting_clause_ids=(),
                    refusal=False,
                    conflicts=(),
                    rationale="Mock rationale",
                    primary_clauses=(),
                    raw_llm_response="Mock answer",
                )

        monkeypatch.setattr(
            "src.pipeline.PolicyQAPipeline.build_from_corpus",
            lambda *args, **kwargs: MockPipeline(),
        )

        result = self.runner.invoke(app, ["ask", "What is the earnings disregard?", "--date", "2026-02-20"])
        assert result.exit_code == 0
        assert captured_dates == [date(2026, 2, 20)]

    def test_cli_no_date_still_works(self, monkeypatch):
        """CLI without --date executes normally."""
        captured_dates = []

        class MockPipeline:
            def ask(self, question, context=None, date=None, top_k=10):
                captured_dates.append(date)
                return GroundedAnswer(
                    question=question,
                    answer_text="Resource limit is $4,000.",
                    status=DecisionStatus.SUPPORTED,
                    citations=(),
                    supporting_clause_ids=(),
                    refusal=False,
                    conflicts=(),
                    rationale="Mock rationale",
                    primary_clauses=(),
                    raw_llm_response="Resource limit is $4,000.",
                )

        monkeypatch.setattr(
            "src.pipeline.PolicyQAPipeline.build_from_corpus",
            lambda *args, **kwargs: MockPipeline(),
        )

        result = self.runner.invoke(app, ["ask", "What is the resource limit for a household?"])
        assert result.exit_code == 0
        assert captured_dates == [None]
        assert "Resource limit is $4,000" in result.output


# ---------------------------------------------------------------------------
# 2. Change Reporting Period Demonstrations (§4.3.2 / §9.1.4)
# ---------------------------------------------------------------------------

class TestReportingChangePipeline:

    def test_reporting_pre_amendment_conflicting(self, temporal_pipeline):
        """
        Pre-amendment date (2026-02-20):
        §4.3.2 (10 calendar days) vs §9.1.4 (30 calendar days) -> CONFLICTING.
        """
        q = "How many calendar days does a recipient have to report a change?"
        answer = temporal_pipeline.ask(q, date="2026-02-20")

        assert answer.status == DecisionStatus.CONFLICTING
        assert answer.refusal is True
        assert "conflicting provisions" in answer.answer_text
        assert "10 calendar days" in answer.answer_text
        assert "30 calendar days" in answer.answer_text

    def test_reporting_effective_date_aligned(self, temporal_pipeline):
        """
        Effective date (2026-03-01):
        Both §4.3.2 and §9.1.4 are aligned to 14 calendar days -> SUPPORTED.
        """
        q = "How many calendar days does a recipient have to report a change?"
        answer = temporal_pipeline.ask(q, date="2026-03-01")

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert "4.3.2" in answer.supporting_clause_ids
        assert len(answer.citations) > 0
        # Citation points to Amendment
        assert any("Amendment No. 2026-01" in cit for cit in answer.citations)

    def test_reporting_post_amendment_supported(self, temporal_pipeline):
        """
        Post-amendment date (2026-04-20):
        14 calendar days applies -> SUPPORTED.
        """
        q = "How many calendar days does a recipient have to report a change?"
        answer = temporal_pipeline.ask(q, date="2026-04-20")

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert "4.3.2" in answer.supporting_clause_ids


# ---------------------------------------------------------------------------
# 3. Earnings Disregard Demonstrations (§6.4.1(a))
# ---------------------------------------------------------------------------

class TestEarningsDisregardPipeline:

    def test_earnings_disregard_pre_amendment(self, temporal_pipeline):
        """
        Pre-amendment date (2026-02-20):
        §6.4.1(a) resolves to original value of $120 per month.
        """
        # Resolve via temporal filter on 6.4.1
        orig_clause = next(c for c in temporal_pipeline.temporal_filter.resolver._original_clauses if c.clause_id == "6.4.1")
        rr = RetrievalResult(clause=orig_clause, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 2, 20)))

        assert len(filtered) == 1
        assert "$120 per month" in filtered[0].clause.text
        assert filtered[0].clause.source_document == "policy_manual.md"

    def test_earnings_disregard_post_amendment(self, temporal_pipeline):
        """
        Post-amendment date (2026-04-20):
        §6.4.1(a) resolves to amended value of $175 per month.
        """
        orig_clause = next(c for c in temporal_pipeline.temporal_filter.resolver._original_clauses if c.clause_id == "6.4.1")
        rr = RetrievalResult(clause=orig_clause, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 4, 20)))

        assert len(filtered) == 1
        assert "$175 per month" in filtered[0].clause.text
        assert filtered[0].clause.source_document == "Amendment No. 2026-01.md"


# ---------------------------------------------------------------------------
# 4. Boundary Date Tests
# ---------------------------------------------------------------------------

class TestBoundaryDatesPipeline:

    def test_boundary_day_before_amendment(self, temporal_pipeline):
        """2026-02-28 uses original policy ($120)."""
        orig_clause = next(c for c in temporal_pipeline.temporal_filter.resolver._original_clauses if c.clause_id == "6.4.1")
        rr = RetrievalResult(clause=orig_clause, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 2, 28)))
        assert "$120 per month" in filtered[0].clause.text

    def test_boundary_effective_day(self, temporal_pipeline):
        """2026-03-01 uses amended policy ($175)."""
        orig_clause = next(c for c in temporal_pipeline.temporal_filter.resolver._original_clauses if c.clause_id == "6.4.1")
        rr = RetrievalResult(clause=orig_clause, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 3, 1)))
        assert "$175 per month" in filtered[0].clause.text

    def test_boundary_day_after_effective(self, temporal_pipeline):
        """2026-03-02 uses amended policy ($175)."""
        orig_clause = next(c for c in temporal_pipeline.temporal_filter.resolver._original_clauses if c.clause_id == "6.4.1")
        rr = RetrievalResult(clause=orig_clause, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 3, 2)))
        assert "$175 per month" in filtered[0].clause.text


# ---------------------------------------------------------------------------
# 5. Missing Date & Unamended Queries
# ---------------------------------------------------------------------------

class TestMissingDateAndUnamendedQueries:

    def test_date_sensitive_without_date_refuses_safely(self, temporal_pipeline):
        """Date-sensitive query without date refuses and explains that a date is required."""
        q = "How many calendar days does a recipient have to report a change?"
        answer = temporal_pipeline.ask(q, date=None)

        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert "--date" in answer.answer_text or "date parameter" in answer.answer_text
        assert "Amendment No. 2026-01" in answer.answer_text

    def test_unamended_query_without_date_answers_supported(self, temporal_pipeline):
        """Unamended query (e.g. resource limit) succeeds without requiring a date."""
        q = "What is the countable resource limit for a household?"
        answer = temporal_pipeline.ask(q, date=None)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert len(answer.supporting_clause_ids) > 0


# ---------------------------------------------------------------------------
# 6. Citations & New Clause §10.5.3A
# ---------------------------------------------------------------------------

class TestCitationsAndNewClause:

    def test_amended_citation_formatting(self, temporal_pipeline):
        """Amended clauses include Amendment provenance in citation string."""
        clause_641 = temporal_pipeline.temporal_filter.resolver._amended_clauses_by_id["6.4.1"]
        cit = format_clause_citation(clause_641)
        assert "§6.4.1" in cit
        assert "Amendment No. 2026-01" in cit

    def test_new_clause_10_5_3A_pre_amendment_not_in_force(self, temporal_pipeline):
        """Before 2026-03-01, §10.5.3A is dropped from filtered results."""
        clause_1053A = temporal_pipeline.temporal_filter.resolver._amended_clauses_by_id["10.5.3A"]
        rr = RetrievalResult(clause=clause_1053A, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 2, 15)))
        assert len(filtered) == 0

    def test_new_clause_10_5_3A_post_amendment_in_force(self, temporal_pipeline):
        """On or after 2026-03-01, §10.5.3A is active and cited."""
        clause_1053A = temporal_pipeline.temporal_filter.resolver._amended_clauses_by_id["10.5.3A"]
        rr = RetrievalResult(clause=clause_1053A, semantic_score=0.9, lexical_score=0.9, combined_score=0.03, sources=("semantic",))
        filtered = temporal_pipeline.temporal_filter.filter([rr], TemporalContext(determination_date=date(2026, 4, 15)))
        assert len(filtered) == 1
        assert "increased the award" in filtered[0].clause.text
        assert "Amendment No. 2026-01" in format_clause_citation(filtered[0].clause)
