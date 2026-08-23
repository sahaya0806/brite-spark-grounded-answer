"""
Grounded Answer Generator — core response synthesis layer.

This layer receives an EvidenceDecision and synthesizes a final user-facing
GroundedAnswer according to strict grounding rules.
"""

from __future__ import annotations

from src.citation.renderer import (
    extract_cited_clause_ids,
    format_clause_citation,
    sanitize_text_citations,
    validate_citations,
)
from src.evidence.models import DecisionStatus, EvidenceDecision
from src.generation.models import GroundedAnswer
from src.generation.prompts import build_grounded_prompt
from src.generation.providers import ChatProvider, OpenAIChatProvider


class GroundedAnswerGenerator:
    """
    Synthesize grounded, cited answers from evaluated policy evidence.

    Parameters
    ----------
    provider:
        Chat completion provider (defaults to OpenAIChatProvider in production).
    """

    def __init__(self, provider: ChatProvider | None = None) -> None:
        self._provider = provider

    def _get_provider(self) -> ChatProvider:
        if self._provider is None:
            self._provider = OpenAIChatProvider()
        return self._provider

    def generate_answer(self, decision: EvidenceDecision) -> GroundedAnswer:
        """
        Generate a GroundedAnswer from an EvidenceDecision.

        Parameters
        ----------
        decision:
            The structured decision produced by the EvidenceEvaluator.

        Returns
        -------
        GroundedAnswer
            Final user-facing answer with verified citations and metadata.
        """
        status = decision.status

        if status == DecisionStatus.SUPPORTED:
            return self._handle_supported(decision)
        elif status == DecisionStatus.CONFLICTING:
            return self._handle_conflicting(decision)
        else:
            return self._handle_insufficient(decision)

    def _handle_supported(self, decision: EvidenceDecision) -> GroundedAnswer:
        """Generate a grounded plain-language answer using ChatProvider."""
        clauses = decision.primary_clauses
        if not clauses:
            # Safety fallback: if no primary clauses exist, refuse
            return self._handle_insufficient(decision)

        provider = self._get_provider()
        messages = build_grounded_prompt(decision.question, clauses)
        raw_text = provider.generate_chat(messages, temperature=0.0)

        # Citation validation & sanitization
        allowed_ids = set(decision.supporting_clause_ids)
        cited_ids = extract_cited_clause_ids(raw_text)
        valid_cited_ids = validate_citations(cited_ids, allowed_ids)
        sanitized_text = sanitize_text_citations(raw_text, allowed_ids)

        # If LLM omitted citation tags, append deterministic citation tags
        if not valid_cited_ids:
            tags = " ".join(f"[§{cid}]" for cid in decision.supporting_clause_ids)
            answer_text = f"{sanitized_text} {tags}".strip()
            final_cids = decision.supporting_clause_ids
        else:
            answer_text = sanitized_text
            final_cids = tuple(valid_cited_ids)

        citations = tuple(format_clause_citation(c) for c in clauses)

        return GroundedAnswer(
            question=decision.question,
            answer_text=answer_text,
            status=DecisionStatus.SUPPORTED,
            citations=citations,
            supporting_clause_ids=final_cids,
            refusal=False,
            conflicts=(),
            rationale=decision.rationale,
            primary_clauses=clauses,
            raw_llm_response=raw_text,
        )

    def _handle_insufficient(self, decision: EvidenceDecision) -> GroundedAnswer:
        """Construct a deterministic refusal explaining why evidence is insufficient."""
        parts = ["The policy manual does not provide enough information to answer this question reliably."]

        if decision.primary_clauses:
            top = decision.primary_clauses[0]
            parts.append(
                f"The closest relevant provision is §{top.clause_id} ({top.section_title}), "
                f"but it does not definitively settle the question."
            )

        if decision.missing_information:
            parts.append(f"Details: {decision.missing_information}")

        parts.append("Please consult the appropriate policy administrator.")
        answer_text = " ".join(parts)

        citations = tuple(format_clause_citation(c) for c in decision.primary_clauses)

        return GroundedAnswer(
            question=decision.question,
            answer_text=answer_text,
            status=DecisionStatus.INSUFFICIENT,
            citations=citations,
            supporting_clause_ids=decision.supporting_clause_ids,
            refusal=True,
            conflicts=(),
            rationale=decision.rationale,
            primary_clauses=decision.primary_clauses,
            raw_llm_response=None,
        )

    def _handle_conflicting(self, decision: EvidenceDecision) -> GroundedAnswer:
        """Construct a deterministic conflict report identifying both provisions."""
        parts = ["The policy manual contains conflicting provisions for this question."]

        for cd in decision.conflict_details:
            parts.append(
                f"§{cd.clause_a.clause_id} states {cd.value_a!r} ({cd.clause_a.section_title}) while "
                f"§{cd.clause_b.clause_id} states {cd.value_b!r} ({cd.clause_b.section_title})."
            )

        parts.append(
            "The manual does not provide enough information for the system to safely determine "
            "which provision governs. Please consult the appropriate policy administrator."
        )
        answer_text = " ".join(parts)

        citations = tuple(format_clause_citation(c) for c in decision.primary_clauses)

        return GroundedAnswer(
            question=decision.question,
            answer_text=answer_text,
            status=DecisionStatus.CONFLICTING,
            citations=citations,
            supporting_clause_ids=decision.supporting_clause_ids,
            refusal=True,
            conflicts=decision.conflict_details,
            rationale=decision.rationale,
            primary_clauses=decision.primary_clauses,
            raw_llm_response=None,
        )
