"""
Integration and end-to-end pipeline tests for Milestone 6.

Uses the real policy manual (data/raw/policy_manual.md) with FakeEmbeddingProvider
and FakeChatProvider for deterministic, offline verification.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from typer.testing import CliRunner

from src.app import app
from src.evidence.models import DecisionStatus
from src.generation.models import GroundedAnswer
from src.generation.providers import FakeChatProvider
from src.pipeline import PolicyQAPipeline
from src.retrieval.embeddings import FakeEmbeddingProvider

REAL_CORPUS = Path("data/raw/policy_manual.md")


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
