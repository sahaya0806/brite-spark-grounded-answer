"""
Evidence evaluation data models.

These models represent the output of the evidence evaluation layer.
They are the contract between evidence evaluation and answer generation.

IMPORTANT:
- These models do NOT contain generated natural-language answers.
- These models do NOT contain final citation rendering.
- These models provide structured evidence decisions for the next milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.ingestion.parser import PolicyClause
from src.retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Decision status
# ---------------------------------------------------------------------------

class DecisionStatus(str, Enum):
    """
    The three possible evidence decisions.

    SUPPORTED
        The retrieved evidence contains sufficient policy text to support
        a substantive answer without unsupported assumptions.

    INSUFFICIENT
        The retrieved evidence is relevant or potentially relevant, but
        the manual does not provide enough reliable information to answer
        the question definitively.  The conservative default when evidence
        is ambiguous.

    CONFLICTING
        The retrieved evidence contains two or more applicable policy
        clauses that materially disagree about the answer to the same
        question.  Neither should be silently preferred.
    """

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"


# ---------------------------------------------------------------------------
# Evidence item
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceItem:
    """
    One piece of policy evidence considered by the evaluator.

    Wraps a ``RetrievalResult`` with additional evaluation-time metadata.

    Attributes
    ----------
    result:
        The underlying ``RetrievalResult`` (contains the ``PolicyClause``
        and retrieval scores).
    relevance_score:
        Float in [0, 1].  Evaluator's assessment of how relevant this
        clause is to the question, derived from retrieval scores and
        lexical signal matching.  NOT answer confidence.
    support_signals:
        Tuple of textual descriptions of signals that suggest this clause
        supports the question.  Empty if none detected.
    gap_signals:
        Tuple of textual descriptions of signals that suggest this clause
        does NOT fully answer the question (e.g. unresolved cross-reference,
        off-topic reference, missing requested condition).
    unresolved_cross_refs:
        Cross-references in this clause that were NOT present among the
        retrieved evidence.  These are potential gaps.
    """

    result: RetrievalResult
    relevance_score: float
    support_signals: tuple[str, ...]
    gap_signals: tuple[str, ...]
    unresolved_cross_refs: tuple[str, ...]

    @property
    def clause(self) -> PolicyClause:
        """Convenience accessor for the underlying PolicyClause."""
        return self.result.clause


# ---------------------------------------------------------------------------
# Conflict detail
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConflictDetail:
    """
    Describes a detected conflict between two evidence clauses.

    Attributes
    ----------
    clause_a:
        First conflicting clause.
    clause_b:
        Second conflicting clause.
    conflict_type:
        Short description of the conflict type, e.g.
        ``"competing_numeric_value"``, ``"competing_duration"``,
        ``"opposing_obligation"``.
    value_a:
        The specific conflicting value/text found in clause_a.
    value_b:
        The specific conflicting value/text found in clause_b.
    explanation:
        Human-readable explanation of why the conflict was detected.
        Used by the answer-generation layer in the next milestone.
    """

    clause_a: PolicyClause
    clause_b: PolicyClause
    conflict_type: str
    value_a: str
    value_b: str
    explanation: str


# ---------------------------------------------------------------------------
# Evidence decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceDecision:
    """
    The structured output of the evidence evaluator.

    This is the contract passed to the answer-generation layer (Milestone 6).

    Attributes
    ----------
    status:
        One of SUPPORTED / INSUFFICIENT / CONFLICTING.
    question:
        The original user question, preserved unchanged.
    evidence:
        All evidence items considered, ordered by relevance_score descending.
    rationale:
        Short internal explanation of how the decision was reached.
        NOT a final user-facing message.
    support_score:
        Aggregate support score in [0, 1] across all evidence items.
        Reflects evidence quality, not answer correctness.
    primary_clauses:
        The clause(s) most directly supporting the decision.
        For SUPPORTED: the supporting clauses.
        For CONFLICTING: the conflicting clauses.
        For INSUFFICIENT: the most relevant clauses found (may be empty).
    conflict_details:
        Details of detected conflicts.  Empty unless status == CONFLICTING.
    missing_information:
        Description of what the manual appears to be missing, if applicable.
        Empty string when not applicable.
    recommended_action:
        Structured hint for the answer-generation layer about what to do.
        E.g. ``"surface_conflict"``, ``"refuse"``, ``"generate_answer"``.
    """

    status: DecisionStatus
    question: str
    evidence: tuple[EvidenceItem, ...]
    rationale: str
    support_score: float
    primary_clauses: tuple[PolicyClause, ...]
    conflict_details: tuple[ConflictDetail, ...]
    missing_information: str
    recommended_action: str

    @property
    def supporting_clause_ids(self) -> tuple[str, ...]:
        """Clause IDs of all primary supporting/conflicting clauses."""
        return tuple(c.clause_id for c in self.primary_clauses)
