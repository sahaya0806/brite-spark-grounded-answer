"""
Answer generation data models.

These models represent the final grounded output delivered to the CLI or caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.evidence.models import ConflictDetail, DecisionStatus
from src.ingestion.parser import PolicyClause


@dataclass(frozen=True)
class GroundedAnswer:
    """
    The structured final result produced by the GroundedAnswerGenerator.

    Attributes
    ----------
    question:
        The original user query.
    answer_text:
        The user-facing plain-language response (grounded answer, refusal,
        or conflict report).
    status:
        The evidence decision status (SUPPORTED, INSUFFICIENT, CONFLICTING).
    citations:
        Formatted citations for all supporting/conflicting clauses.
    supporting_clause_ids:
        Clause IDs of all supporting or conflicting clauses.
    refusal:
        True if the question could not be supported (INSUFFICIENT or CONFLICTING).
    conflicts:
        Details of any detected policy conflicts.
    rationale:
        Internal reasoning from the evidence evaluation stage.
    primary_clauses:
        Authoritative PolicyClause records used to produce the answer.
    raw_llm_response:
        Raw LLM text response if an LLM was invoked, or None.
    """

    question: str
    answer_text: str
    status: DecisionStatus
    citations: tuple[str, ...]
    supporting_clause_ids: tuple[str, ...]
    refusal: bool
    conflicts: tuple[ConflictDetail, ...]
    rationale: str
    primary_clauses: tuple[PolicyClause, ...]
    raw_llm_response: str | None = None
