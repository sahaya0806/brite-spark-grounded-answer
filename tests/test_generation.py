"""
Unit tests for Milestone 6 — Grounded Answer Generation.

All tests run completely offline using FakeChatProvider and synthetic models.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.citation.renderer import (
    extract_cited_clause_ids,
    format_clause_citation,
    format_short_citation,
    sanitize_text_citations,
    validate_citations,
)
from src.evidence.models import (
    ConflictDetail,
    DecisionStatus,
    EvidenceDecision,
    EvidenceItem,
)
from src.generation.generator import GroundedAnswerGenerator
from src.generation.models import GroundedAnswer
from src.generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from src.generation.providers import (
    ChatProvider,
    FakeChatProvider,
    OpenAIChatProvider,
)
from src.ingestion.parser import PolicyClause
from src.retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _clause(
    cid: str,
    text: str,
    section_id: str = "1.1",
    part_id: str = "1",
    start_line: int = 10,
    end_line: int = 10,
) -> PolicyClause:
    return PolicyClause(
        clause_id=cid,
        part_id=part_id,
        part_title=f"Part {part_id}",
        section_id=section_id,
        section_title=f"{section_id} Title",
        text=text,
        sub_items=(),
        cross_references=(),
        source_path=Path("/fake/manual.md"),
        start_line=start_line,
        end_line=end_line,
    )


def _decision(
    status: DecisionStatus,
    question: str = "Test question?",
    primary_clauses: tuple[PolicyClause, ...] = (),
    conflict_details: tuple[ConflictDetail, ...] = (),
    missing_info: str = "",
) -> EvidenceDecision:
    return EvidenceDecision(
        status=status,
        question=question,
        evidence=(),
        rationale="Test rationale.",
        support_score=0.8 if status == DecisionStatus.SUPPORTED else 0.0,
        primary_clauses=primary_clauses,
        conflict_details=conflict_details,
        missing_information=missing_info,
        recommended_action="generate_answer" if status == DecisionStatus.SUPPORTED else "refuse",
    )


# ---------------------------------------------------------------------------
# 1. Chat Providers
# ---------------------------------------------------------------------------

class TestChatProviders:

    def test_fake_chat_provider_implements_protocol(self):
        provider = FakeChatProvider()
        assert isinstance(provider, ChatProvider)

    def test_fake_chat_provider_canned_response(self):
        provider = FakeChatProvider(canned_response="Canned answer.")
        resp = provider.generate_chat([{"role": "user", "content": "hello"}])
        assert resp == "Canned answer."
        assert len(provider.call_history) == 1

    def test_fake_chat_provider_custom_responder(self):
        provider = FakeChatProvider(responder=lambda msgs: f"Echo: {msgs[0]['content']}")
        resp = provider.generate_chat([{"role": "user", "content": "custom"}])
        assert resp == "Echo: custom"

    def test_fake_chat_provider_extracts_citations_from_prompt(self):
        provider = FakeChatProvider()
        prompt = [{"role": "user", "content": "Question? EVIDENCE: [§2.4.1] text here"}]
        resp = provider.generate_chat(prompt)
        assert "[§2.4.1]" in resp

    def test_openai_chat_provider_missing_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIChatProvider()

    def test_openai_chat_provider_custom_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        provider = OpenAIChatProvider(model="custom-gpt")
        assert provider._model == "custom-gpt"


# ---------------------------------------------------------------------------
# 2. Citation Renderer & Validation
# ---------------------------------------------------------------------------

class TestCitationRenderer:

    def test_format_clause_citation_single_line(self):
        c = _clause("4.3.2", "Text", start_line=200, end_line=200)
        assert format_clause_citation(c) == "§4.3.2, line 200"

    def test_format_clause_citation_line_range(self):
        c = _clause("6.4.1", "Text", start_line=300, end_line=320)
        assert format_clause_citation(c) == "§6.4.1, lines 300–320"

    def test_format_short_citation(self):
        c = _clause("4.3.2", "Text")
        assert format_short_citation(c) == "[§4.3.2]"
        assert format_short_citation("9.1.4") == "[§9.1.4]"

    def test_extract_cited_clause_ids(self):
        text = "Under §4.3.2 and [§6.6.1], see also clause 2.4.1 and §4.3.2 again."
        extracted = extract_cited_clause_ids(text)
        assert extracted == ["4.3.2", "6.6.1", "2.4.1"]

    def test_validate_citations_rejects_hallucinated_ids(self):
        cited = ["4.3.2", "99.99", "6.6.1"]
        allowed = {"4.3.2", "6.6.1"}
        valid = validate_citations(cited, allowed)
        assert valid == ["4.3.2", "6.6.1"]
        assert "99.99" not in valid

    def test_sanitize_text_citations_removes_unallowed(self):
        text = "The rule is [§4.3.2] but not [§99.99]."
        sanitized = sanitize_text_citations(text, {"4.3.2"})
        assert "[§4.3.2]" in sanitized
        assert "99.99" not in sanitized


# ---------------------------------------------------------------------------
# 3. Prompt Construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:

    def test_system_prompt_contains_grounding_rules(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "[§<clause_id>]" in SYSTEM_PROMPT
        assert "Never use external knowledge" in SYSTEM_PROMPT

    def test_build_grounded_prompt_includes_evidence(self):
        c1 = _clause("2.4.1", "Countable resources limit is $4,000.", section_id="2.4", part_id="2")
        messages = build_grounded_prompt("What is the resource limit?", [c1])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "What is the resource limit?" in user_content
        assert "[§2.4.1]" in user_content
        assert "Countable resources limit is $4,000." in user_content


# ---------------------------------------------------------------------------
# 4. GroundedAnswerGenerator — SUPPORTED
# ---------------------------------------------------------------------------

class TestGroundedAnswerGeneratorSupported:

    def test_supported_generates_answer_with_citations(self):
        c = _clause("2.4.1", "Countable resources must not exceed $4,000.", start_line=80, end_line=80)
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="What is the resource limit?",
            primary_clauses=(c,),
        )
        fake_llm = FakeChatProvider(canned_response="The resource limit is $4,000 [§2.4.1].")
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)

        assert isinstance(answer, GroundedAnswer)
        assert answer.status == DecisionStatus.SUPPORTED
        assert answer.refusal is False
        assert "$4,000" in answer.answer_text
        assert "[§2.4.1]" in answer.answer_text
        assert answer.supporting_clause_ids == ("2.4.1",)
        assert answer.citations == ("§2.4.1, line 80",)
        assert answer.raw_llm_response == "The resource limit is $4,000 [§2.4.1]."

    def test_supported_appends_citation_if_llm_omits_it(self):
        c = _clause("2.4.1", "Countable resources must not exceed $4,000.")
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="What is the resource limit?",
            primary_clauses=(c,),
        )
        # LLM forgot to include citation tag
        fake_llm = FakeChatProvider(canned_response="The household resource limit is $4,000.")
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)
        assert "[§2.4.1]" in answer.answer_text
        assert answer.supporting_clause_ids == ("2.4.1",)

    def test_supported_strips_hallucinated_citation(self):
        c = _clause("2.4.1", "Countable resources must not exceed $4,000.")
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="What is the resource limit?",
            primary_clauses=(c,),
        )
        # LLM hallucinated an extra citation [§99.99]
        fake_llm = FakeChatProvider(canned_response="Resources limit is $4,000 [§2.4.1] see also [§99.99].")
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)
        assert "[§2.4.1]" in answer.answer_text
        assert "99.99" not in answer.answer_text
        assert answer.supporting_clause_ids == ("2.4.1",)

    def test_supported_multiple_clauses(self):
        c1 = _clause("2.1.2", "Eligibility conditions.", start_line=72, end_line=72)
        c2 = _clause("2.4.1", "Countable resources limit $4,000.", start_line=80, end_line=80)
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="What are eligibility conditions?",
            primary_clauses=(c1, c2),
        )
        fake_llm = FakeChatProvider(canned_response="Conditions include [§2.1.2] and resources [§2.4.1].")
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)
        assert answer.status == DecisionStatus.SUPPORTED
        assert "2.1.2" in answer.supporting_clause_ids
        assert "2.4.1" in answer.supporting_clause_ids
        assert len(answer.citations) == 2


# ---------------------------------------------------------------------------
# 5. GroundedAnswerGenerator — INSUFFICIENT
# ---------------------------------------------------------------------------

class TestGroundedAnswerGeneratorInsufficient:

    def test_insufficient_refuses_without_calling_llm(self):
        dec = _decision(
            status=DecisionStatus.INSUFFICIENT,
            question="What is the policy for full-time students?",
            primary_clauses=(),
            missing_info="No policy clauses found.",
        )
        fake_llm = FakeChatProvider()
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)

        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert "does not provide enough information" in answer.answer_text
        assert "Please consult the appropriate policy administrator." in answer.answer_text
        assert len(fake_llm.call_history) == 0  # Zero LLM calls!

    def test_insufficient_mentions_closest_clause(self):
        c = _clause("7.1.3", "Needs figure for full-time students (see §5.4).", section_id="7.1")
        dec = _decision(
            status=DecisionStatus.INSUFFICIENT,
            question="What is the policy for full-time students?",
            primary_clauses=(c,),
            missing_info="unresolved_cross_refs:§5.4",
        )
        fake_llm = FakeChatProvider()
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)
        assert "§7.1.3" in answer.answer_text
        assert answer.refusal is True
        assert len(fake_llm.call_history) == 0


# ---------------------------------------------------------------------------
# 6. GroundedAnswerGenerator — CONFLICTING
# ---------------------------------------------------------------------------

class TestGroundedAnswerGeneratorConflicting:

    def test_conflicting_surfaces_both_clauses_without_calling_llm(self):
        c1 = _clause("4.3.2", "report within 10 calendar days", section_id="4.3")
        c2 = _clause("9.1.4", "reported within 30 calendar days", section_id="9.1")
        conflict = ConflictDetail(
            clause_a=c1,
            clause_b=c2,
            conflict_type="competing_duration_days",
            value_a="10 calendar days",
            value_b="30 calendar days",
            explanation="Different days for same obligation",
        )
        dec = _decision(
            status=DecisionStatus.CONFLICTING,
            question="How many days to report a change?",
            primary_clauses=(c1, c2),
            conflict_details=(conflict,),
        )
        fake_llm = FakeChatProvider()
        generator = GroundedAnswerGenerator(provider=fake_llm)

        answer = generator.generate_answer(dec)

        assert answer.status == DecisionStatus.CONFLICTING
        assert answer.refusal is True
        assert "conflicting provisions" in answer.answer_text
        assert "§4.3.2" in answer.answer_text
        assert "10 calendar days" in answer.answer_text
        assert "§9.1.4" in answer.answer_text
        assert "30 calendar days" in answer.answer_text
        assert "Please consult the appropriate policy administrator." in answer.answer_text
        assert len(fake_llm.call_history) == 0  # Zero LLM calls!


# ---------------------------------------------------------------------------
# 7. Additional Edge Cases & Robustness
# ---------------------------------------------------------------------------

class TestAnswerGeneratorEdgeCases:

    def test_supported_empty_clauses_falls_back_to_refusal(self):
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="Question?",
            primary_clauses=(),  # empty!
        )
        fake_llm = FakeChatProvider()
        generator = GroundedAnswerGenerator(provider=fake_llm)
        answer = generator.generate_answer(dec)
        assert answer.status == DecisionStatus.INSUFFICIENT
        assert answer.refusal is True
        assert len(fake_llm.call_history) == 0

    def test_empty_llm_response_appends_valid_citation(self):
        c = _clause("2.4.1", "Countable resources limit $4,000.")
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="What is the resource limit?",
            primary_clauses=(c,),
        )
        fake_llm = FakeChatProvider(canned_response="")
        generator = GroundedAnswerGenerator(provider=fake_llm)
        answer = generator.generate_answer(dec)
        assert "[§2.4.1]" in answer.answer_text
        assert answer.supporting_clause_ids == ("2.4.1",)

    def test_malformed_citation_patterns(self):
        text = "Check [§] and [§invalid] and [§1.2.3.4.5] and [§4.3.2]."
        extracted = extract_cited_clause_ids(text)
        assert "4.3.2" in extracted
        valid = validate_citations(extracted, {"4.3.2"})
        assert valid == ["4.3.2"]

    def test_grounded_answer_is_frozen(self):
        c = _clause("2.4.1", "Text")
        dec = _decision(
            status=DecisionStatus.SUPPORTED,
            question="Q",
            primary_clauses=(c,),
        )
        generator = GroundedAnswerGenerator(provider=FakeChatProvider())
        answer = generator.generate_answer(dec)
        with pytest.raises((AttributeError, TypeError)):
            answer.status = DecisionStatus.INSUFFICIENT  # type: ignore[misc]
