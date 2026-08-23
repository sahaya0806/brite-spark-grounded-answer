"""
Integration and end-to-end pipeline tests for Milestone 6.

Uses the real policy manual (data/raw/policy_manual.md) with FakeEmbeddingProvider
and FakeChatProvider for deterministic, offline verification.

Milestone 6 validation fix tests are in TestApparentGapRegressions and
TestConflictRenderingRegressions.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from typer.testing import CliRunner

from src.app import app
from src.evidence.evaluator import EvidenceEvaluator, _is_ref_topically_resolved, _meaningful_query_tokens
from src.evidence.models import DecisionStatus
from src.generation.generator import GroundedAnswerGenerator
from src.generation.models import GroundedAnswer
from src.generation.providers import FakeChatProvider
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import parse_clauses
from src.pipeline import PolicyQAPipeline
from src.retrieval.embeddings import FakeEmbeddingProvider
from src.retrieval.models import RetrievalResult

REAL_CORPUS = Path("data/raw/policy_manual.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_result(clause, sem: float, lex: float) -> RetrievalResult:
    return RetrievalResult(
        clause=clause,
        semantic_score=sem,
        lexical_score=lex,
        combined_score=(sem + lex) / 2,
        sources=frozenset(),
    )


@pytest.fixture(scope="module")
def corpus_clauses():
    doc = load_policy_document(REAL_CORPUS)
    return {c.clause_id: c for c in parse_clauses(doc)}


# ---------------------------------------------------------------------------
# TestApparentGapRegressions
# Directly protects: retrieval relevance ≠ evidence sufficiency
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestApparentGapRegressions:
    """
    Regression suite for the §7.1.3 apparent-gap bug.

    §7.1.3 refers full-time student policy to §5.4.
    §5.4.x clauses cover CARE ALLOWANCES, not full-time students.
    The evaluator must not confuse §5.4.x relevance with §5.4 delegation resolution.
    """

    def test_713_delegates_to_54_not_topically_resolved_by_541(self, corpus_clauses):
        """
        §5.4.1 (care allowances) must NOT count as a topical resolution of §7.1.3's
        delegation to §5.4 for full-time student policy.
        """
        c713 = corpus_clauses["7.1.3"]
        c541 = corpus_clauses["5.4.1"]
        c542 = corpus_clauses["5.4.2"]

        query = "What is the policy for full-time students?"
        query_tokens = _meaningful_query_tokens(query)

        results = [
            _make_retrieval_result(c713, 0.9, 0.8),
            _make_retrieval_result(c541, 0.7, 0.1),
            _make_retrieval_result(c542, 0.6, 0.1),
        ]
        retrieved_ids = frozenset(r.clause.clause_id for r in results)

        # §5.4 should NOT be considered resolved by §5.4.1 (care allowances)
        is_resolved = _is_ref_topically_resolved("5.4", retrieved_ids, results, query_tokens)
        assert is_resolved is False, (
            "§5.4.1 (care allowances) must not count as a resolution of "
            "§7.1.3's delegation of full-time student policy to §5.4"
        )

    def test_713_delegates_to_54_evaluator_returns_insufficient(self, corpus_clauses):
        """
        When §7.1.3 delegates full-time student policy to §5.4 and only care-allowance
        clauses (§5.4.1, §5.4.2) are retrieved, EvidenceEvaluator must return INSUFFICIENT.
        """
        c713 = corpus_clauses["7.1.3"]
        c541 = corpus_clauses["5.4.1"]
        c542 = corpus_clauses["5.4.2"]
        c323 = corpus_clauses["3.2.3"]

        results = [
            _make_retrieval_result(c713, 0.9, 0.8),
            _make_retrieval_result(c541, 0.7, 0.1),
            _make_retrieval_result(c542, 0.6, 0.1),
            _make_retrieval_result(c323, 0.5, 0.3),
        ]

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("What is the policy for full-time students?", results)

        assert decision.status == DecisionStatus.INSUFFICIENT, (
            "Retrieving care-allowance clauses (§5.4.1, §5.4.2) alongside §7.1.3 "
            "must NOT upgrade the decision to SUPPORTED. "
            "Retrieval relevance ≠ evidence sufficiency."
        )

    def test_541_does_not_resolve_713_student_delegation(self, corpus_clauses):
        """
        §5.4.1 (care allowances) must not be treated as resolving §7.1.3's
        delegation of full-time student policy to §5.4, even when §5.4.1 is
        retrieved alongside §7.1.3.

        The invariant: topical resolution of a delegation requires that the
        retrieved sub-clause actually discusses the delegated topic (full-time
        students), not a different topic (care allowances).
        """
        c713 = corpus_clauses["7.1.3"]
        c541 = corpus_clauses["5.4.1"]

        # §7.1.3 creates the delegation: it says 'see §5.4' for students
        # §5.4.1 is about care allowances — it should NOT resolve that delegation
        results = [
            _make_retrieval_result(c713, 0.9, 0.8),
            _make_retrieval_result(c541, 0.8, 0.7),
        ]

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("What is the policy for full-time students?", results)

        # With only §7.1.3 (delegating) and §5.4.1 (wrong topic),
        # the delegation must remain unresolved → INSUFFICIENT
        assert decision.status == DecisionStatus.INSUFFICIENT, (
            "§5.4.1 (care allowances) must not resolve §7.1.3's delegation "
            "of full-time student policy to §5.4. Got: "
            f"{decision.status} — {decision.rationale}"
        )


    def test_713_missing_information_mentions_54(self, corpus_clauses):
        """The INSUFFICIENT result for the student question must cite the gap to §5.4."""
        c713 = corpus_clauses["7.1.3"]

        results = [_make_retrieval_result(c713, 0.9, 0.8)]

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("What is the policy for full-time students?", results)

        assert decision.status == DecisionStatus.INSUFFICIENT
        # Rationale should mention the unresolved delegation to §5.4
        assert "5.4" in decision.rationale or "5.4" in decision.missing_information

    def test_llm_cannot_override_insufficient_decision(self, corpus_clauses):
        """
        The GroundedAnswerGenerator must produce a refusal when EvidenceDecision is
        INSUFFICIENT, regardless of what a chat provider might return.
        """
        c713 = corpus_clauses["7.1.3"]
        c541 = corpus_clauses["5.4.1"]

        results = [
            _make_retrieval_result(c713, 0.9, 0.8),
            _make_retrieval_result(c541, 0.7, 0.1),
        ]

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("What is the policy for full-time students?", results)
        assert decision.status == DecisionStatus.INSUFFICIENT

        # Even a chat provider that would return a confident-looking answer
        # must not change the outcome
        fake_llm = FakeChatProvider(
            canned_response="Full-time students are covered under §7.1.3. [§7.1.3]"
        )
        generator = GroundedAnswerGenerator(provider=fake_llm)
        answer = generator.generate_answer(decision)

        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert "does not provide enough information" in answer.answer_text
        # LLM must not have been called for INSUFFICIENT decisions
        assert len(fake_llm.call_history) == 0

    def test_final_answer_for_student_question_is_refusal(self, corpus_clauses):
        """Final GroundedAnswer for student question must be a refusal."""
        c713 = corpus_clauses["7.1.3"]

        results = [_make_retrieval_result(c713, 0.9, 0.8)]

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate("What is the policy for full-time students?", results)
        generator = GroundedAnswerGenerator(provider=FakeChatProvider())
        answer = generator.generate_answer(decision)

        assert answer.refusal is True
        assert answer.status == DecisionStatus.INSUFFICIENT
        assert "Please consult the appropriate policy administrator." in answer.answer_text

    def test_7_3_ref_also_flagged_as_unresolved_when_not_retrieved(self, corpus_clauses):
        """
        §7.1.3 also cross-references §7.3. When §7.3.x is not retrieved, that
        reference should also be unresolved (though it is secondary to the §5.4 gap).
        """
        c713 = corpus_clauses["7.1.3"]
        results = [_make_retrieval_result(c713, 0.9, 0.8)]
        retrieved_ids = frozenset(r.clause.clause_id for r in results)
        query_tokens = _meaningful_query_tokens("What is the policy for full-time students?")

        # §7.3 not retrieved → should be unresolved
        is_resolved = _is_ref_topically_resolved("7.3", retrieved_ids, results, query_tokens)
        assert is_resolved is False


# ---------------------------------------------------------------------------
# TestConflictRenderingRegressions
# Protects against duplicate "10 calendar days, 10 calendar days" rendering
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestConflictRenderingRegressions:
    """
    Regression suite for duplicate numeric value rendering in conflict reports.

    §4.3.2 says "10 calendar days" twice in its text.
    The conflict report must render it once, not twice.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        return PolicyQAPipeline.build_from_corpus(
            corpus_path=REAL_CORPUS,
            embedding_provider=FakeEmbeddingProvider(dim=64),
            chat_provider=FakeChatProvider(),
        )

    def test_conflict_does_not_contain_duplicate_days_text(self, corpus_clauses):
        """
        The conflict report for §4.3.2 vs §9.1.4 must not contain
        '10 calendar days, 10 calendar days' (duplicate).
        """
        from src.evidence.contradiction import detect_conflicts
        from src.evidence.evaluator import EvidenceEvaluator

        c432 = corpus_clauses["4.3.2"]
        c914 = corpus_clauses["9.1.4"]

        results = [
            _make_retrieval_result(c432, 0.9, 0.8),
            _make_retrieval_result(c914, 0.8, 0.7),
        ]
        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate(
            "How many days does a recipient have to report a change?", results
        )

        assert decision.status == DecisionStatus.CONFLICTING
        for conflict in decision.conflict_details:
            assert "10 calendar days, 10 calendar days" not in conflict.value_a, (
                "value_a must not contain duplicate day text. "
                f"Got: {conflict.value_a!r}"
            )
            assert "10 calendar days, 10 calendar days" not in conflict.value_b

    def test_conflict_value_a_is_10_calendar_days(self, corpus_clauses):
        """The conflict for §4.3.2 must render as '10 calendar days', not duplicated."""
        from src.evidence.evaluator import EvidenceEvaluator

        c432 = corpus_clauses["4.3.2"]
        c914 = corpus_clauses["9.1.4"]

        results = [
            _make_retrieval_result(c432, 0.9, 0.8),
            _make_retrieval_result(c914, 0.8, 0.7),
        ]
        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate(
            "How many days does a recipient have to report a change?", results
        )

        assert decision.status == DecisionStatus.CONFLICTING
        found_432_conflict = None
        for cd in decision.conflict_details:
            if cd.clause_a.clause_id == "4.3.2" or cd.clause_b.clause_id == "4.3.2":
                found_432_conflict = cd
                break

        assert found_432_conflict is not None, "No conflict found involving §4.3.2"
        if found_432_conflict.clause_a.clause_id == "4.3.2":
            assert found_432_conflict.value_a == "10 calendar days"
        else:
            assert found_432_conflict.value_b == "10 calendar days"

    def test_existing_contradiction_still_conflicting(self, pipeline):
        """Regression: §4.3.2 vs §9.1.4 contradiction must still be CONFLICTING."""
        q = "How many days does a recipient have to report a change?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.CONFLICTING
        assert answer.refusal is True
        assert "10 calendar days" in answer.answer_text
        assert "30 calendar days" in answer.answer_text
        # Must not contain the duplicated string
        assert "10 calendar days, 10 calendar days" not in answer.answer_text

    def test_supported_applicant_information_not_regressed(self, pipeline):
        """Regression: SUPPORTED case must remain SUPPORTED after the fix."""
        q = "What information must an applicant provide?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert len(answer.citations) > 0

    def test_external_concept_refusal_federal_district_court(self, pipeline):
        """
        An out-of-scope query referencing concepts not in the manual
        ('Federal District Court') must be refused as INSUFFICIENT,
        preventing the LLM from extrapolating from internal appeal rules.
        """
        q = "Can an applicant appeal a decision directly to the Federal District Court?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert "does not provide enough information" in answer.answer_text

    def test_resource_limit_no_false_conflict(self, pipeline):
        """
        Resource limit question must not produce a false CONFLICTING decision
        against monthly income thresholds (§6.6.1).
        """
        q = "What is the countable resource limit for a household?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert answer.conflicts == ()

    def test_supervisor_referral_no_false_conflict(self, pipeline):
        """
        Supervisor referral question must not produce a false CONFLICTING decision
        between suspension continuation (60 days, §10.2.3) and determination (90 days, §8.3.3).
        """
        q = "When must a determination be referred to a supervisor?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert answer.conflicts == ()

    def test_identity_verification_no_false_conflict(self, pipeline):
        """
        Identity verification question must not produce a false CONFLICTING decision
        between evidence deadlines (14 days, §8.2.3) and county presence (30 days, §3.3.1).
        """
        q = "What information must an applicant provide to verify identity?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert answer.conflicts == ()



# ---------------------------------------------------------------------------
# Original E2E suite — must not regress
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestPolicyQAPipelineRealCorpus:

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Builds an end-to-end pipeline on the real corpus with fake providers."""
        emb_provider = FakeEmbeddingProvider(dim=64)
        chat_provider = FakeChatProvider()
        return PolicyQAPipeline.build_from_corpus(
            corpus_path=REAL_CORPUS,
            embedding_provider=emb_provider,
            chat_provider=chat_provider,
        )

    def test_e2e_known_contradiction(self, pipeline):
        q = "How many calendar days does a recipient have to report a change?"
        answer = pipeline.ask(q)

        assert isinstance(answer, GroundedAnswer)
        assert answer.status == DecisionStatus.CONFLICTING
        assert answer.refusal is True
        assert "conflicting provisions" in answer.answer_text
        assert "§4.3.2" in answer.answer_text
        assert "§9.1.4" in answer.answer_text
        assert "10 calendar days" in answer.answer_text
        assert "30 calendar days" in answer.answer_text
        assert "4.3.2" in answer.supporting_clause_ids
        assert "9.1.4" in answer.supporting_clause_ids
        # No duplicate rendering
        assert "10 calendar days, 10 calendar days" not in answer.answer_text

    def test_e2e_known_gap_student_policy(self, pipeline):
        q = "What is the policy for full-time students?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert "does not provide enough information" in answer.answer_text
        assert "§7.1.3" in answer.answer_text
        assert "Please consult the appropriate policy administrator." in answer.answer_text

    def test_e2e_known_supported_resource_limit(self, pipeline):
        q = "What is the resource limit for a household?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert len(answer.citations) > 0
        assert len(answer.supporting_clause_ids) > 0
        for cid in answer.supporting_clause_ids:
            assert f"[§{cid}]" in answer.answer_text

    def test_e2e_known_supported_income_threshold(self, pipeline):
        q = "What is the countable income threshold for eligibility?"
        answer = pipeline.ask(q)

        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert len(answer.citations) > 0


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------

class TestCLICommands:

    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_info_command(self):
        result = self.runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Milestone 6" in result.output
        assert "The Grounded Answer" in result.output

    def test_cli_ask_command_missing_corpus(self, tmp_path):
        missing_file = tmp_path / "nonexistent.md"
        result = self.runner.invoke(app, ["ask", "What is the limit?", "--corpus", str(missing_file)])
        assert result.exit_code != 0
        assert "Error: Policy corpus not found" in result.output

    @pytest.mark.skipif(
        not REAL_CORPUS.exists(),
        reason="Real corpus not present",
    )
    def test_cli_ask_with_mocked_pipeline(self, monkeypatch):
        from src.generation.providers import FakeChatProvider
        from src.retrieval.embeddings import FakeEmbeddingProvider

        fake_pipeline = PolicyQAPipeline.build_from_corpus(
            REAL_CORPUS,
            embedding_provider=FakeEmbeddingProvider(dim=64),
            chat_provider=FakeChatProvider(),
        )
        monkeypatch.setattr(
            "src.pipeline.PolicyQAPipeline.build_from_corpus",
            lambda *args, **kwargs: fake_pipeline,
        )

        result = self.runner.invoke(app, ["ask", "What is the resource limit for a household?"])
        assert result.exit_code == 0
        assert "Status: [SUPPORTED]" in result.output
        assert "Answer:" in result.output
        assert "Citations:" in result.output
