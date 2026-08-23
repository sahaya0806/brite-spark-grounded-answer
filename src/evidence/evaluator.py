"""
Evidence evaluator — core decision logic.

Decision flow
-------------
1.  Receive the question and retrieved evidence (list[RetrievalResult]).
2.  Build EvidenceItem objects: compute relevance scores and extract
    support/gap signals for each retrieved clause.
3.  Compute aggregate support score across all evidence items.
4.  Detect conflicts between evidence items (contradiction check).
5.  Apply decision rules (in order of priority):
    a. If no relevant evidence: INSUFFICIENT.
    b. If material conflict detected: CONFLICTING.
    c. If aggregate support score >= threshold AND at least one
       strong evidence item without unresolved cross-references: SUPPORTED.
    d. Otherwise: INSUFFICIENT (conservative default).

Conservative policy
-------------------
The evaluator defaults to INSUFFICIENT when evidence is ambiguous.
A false refusal is undesirable but recoverable.
A confident unsupported answer undermines the system's trustworthiness.

This is the principal architectural trade-off for this challenge:
we set the evidence bar higher than "top retrieval result exists."

Thresholds (configurable via EvidenceConfig)
--------------------------------------------
- min_relevance:       minimum per-item score to count as relevant (0.15)
- min_support:         aggregate score threshold for SUPPORTED (0.35)
- strong_item_score:   individual item score for "strong evidence" (0.40)
- conflict_min_relevance: minimum per-item score to trigger conflict (0.10)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.evidence.contradiction import detect_conflicts
from src.evidence.models import (
    ConflictDetail,
    DecisionStatus,
    EvidenceDecision,
    EvidenceItem,
)
from src.evidence.scoring import (
    compute_relevance_score,
    extract_gap_signals,
    extract_support_signals,
)
from src.retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvidenceConfig:
    """
    Thresholds and parameters for the evidence evaluator.

    All fields are configurable so the decision boundary can be tuned
    and tested without code changes.

    Attributes
    ----------
    min_relevance:
        Minimum relevance_score for an evidence item to count toward
        the aggregate support calculation.  Items below this are ignored.
    min_support:
        Aggregate support score (weighted average of relevant items'
        relevance scores) required for a SUPPORTED decision.
        Higher = more conservative.
    strong_item_score:
        An individual evidence item with relevance_score >= this value
        is considered "strong evidence."  At least one strong item is
        required for SUPPORTED.
    conflict_min_relevance:
        Minimum per-item relevance to participate in conflict detection.
    """
    min_relevance: float = 0.15
    min_support: float = 0.35
    strong_item_score: float = 0.40
    conflict_min_relevance: float = 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EvidenceEvaluator:
    """
    Evaluate whether retrieved policy evidence supports a definitive answer.

    Usage
    -----
    ::

        evaluator = EvidenceEvaluator()
        decision = evaluator.evaluate(question, retrieval_results)
        print(decision.status)      # SUPPORTED / INSUFFICIENT / CONFLICTING

    Parameters
    ----------
    config:
        Evaluation thresholds.  Defaults are conservative.
    """

    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self._config = config or EvidenceConfig()

    def evaluate(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
    ) -> EvidenceDecision:
        """
        Evaluate the evidence for *question*.

        Parameters
        ----------
        question:
            The original user question, unchanged.
        retrieval_results:
            Retrieved policy clauses from the hybrid retriever.

        Returns
        -------
        EvidenceDecision
            Structured decision with status, rationale, and evidence details.
        """
        cfg = self._config
        query_tokens = _meaningful_query_tokens(question)

        # Deduplicate retrieval results by clause_id preserving order
        seen_ids: set[str] = set()
        deduped_results: list[RetrievalResult] = []
        for r in retrieval_results:
            if r.clause.clause_id not in seen_ids:
                seen_ids.add(r.clause.clause_id)
                deduped_results.append(r)

        retrieved_ids = frozenset(r.clause.clause_id for r in deduped_results)

        # --- Step 1: Build evidence items ---
        evidence_items: list[EvidenceItem] = []
        for result in deduped_results:
            support = extract_support_signals(result.clause, query_tokens)
            gaps = extract_gap_signals(result.clause, retrieved_ids, query_tokens)
            unresolved = tuple(
                ref for ref in result.clause.cross_references
                if _ref_id(ref) not in retrieved_ids
                and not any(
                    rid.startswith(_ref_id(ref) + ".")
                    for rid in retrieved_ids
                )
            )
            relevance = compute_relevance_score(result, support)
            evidence_items.append(EvidenceItem(
                result=result,
                relevance_score=relevance,
                support_signals=support,
                gap_signals=gaps,
                unresolved_cross_refs=unresolved,
            ))

        # Sort by relevance descending
        evidence_items.sort(key=lambda e: e.relevance_score, reverse=True)

        # --- Step 2: Filter to relevant items ---
        relevant_items = [
            item for item in evidence_items
            if item.relevance_score >= cfg.min_relevance
        ]

        # --- Step 3: No evidence at all ---
        if not evidence_items:
            return _make_decision(
                status=DecisionStatus.INSUFFICIENT,
                question=question,
                evidence=tuple(evidence_items),
                rationale="No evidence was retrieved.",
                support_score=0.0,
                primary_clauses=(),
                conflict_details=(),
                missing_information="No policy clauses were retrieved for this question.",
                recommended_action="refuse",
            )

        # --- Step 4: Detect conflicts ---
        conflicts = detect_conflicts(
            evidence_items,
            conflict_min_relevance=cfg.conflict_min_relevance,
        )

        # --- Step 5: Aggregate support score ---
        if relevant_items:
            support_score = sum(
                item.relevance_score for item in relevant_items
            ) / len(relevant_items)
        else:
            support_score = 0.0

        # --- Step 6: Apply decision rules ---

        # 6a. No relevant evidence above threshold
        if not relevant_items:
            return _make_decision(
                status=DecisionStatus.INSUFFICIENT,
                question=question,
                evidence=tuple(evidence_items),
                rationale=(
                    f"Retrieved evidence but no item scored above "
                    f"relevance threshold ({cfg.min_relevance:.2f}). "
                    f"Top item scored {evidence_items[0].relevance_score:.3f}."
                ),
                support_score=support_score,
                primary_clauses=tuple(i.clause for i in evidence_items[:3]),
                conflict_details=(),
                missing_information=(
                    "Retrieved clauses appear peripheral rather than "
                    "directly addressing the question."
                ),
                recommended_action="refuse",
            )

        # 6b. Material conflict
        if conflicts:
            conflicting_clauses = tuple(
                _unique_clauses_from_conflicts(conflicts)
            )
            return _make_decision(
                status=DecisionStatus.CONFLICTING,
                question=question,
                evidence=tuple(evidence_items),
                rationale=(
                    f"Detected {len(conflicts)} conflicting provision(s) "
                    f"in the retrieved evidence. "
                    f"The manual appears to contain contradictory rules "
                    f"for this question."
                ),
                support_score=support_score,
                primary_clauses=conflicting_clauses,
                conflict_details=tuple(conflicts),
                missing_information="",
                recommended_action="surface_conflict",
            )

        # 6c. SUPPORTED: strong evidence, no material conflict
        strong_items = [
            item for item in relevant_items
            if item.relevance_score >= cfg.strong_item_score
            and not item.unresolved_cross_refs
        ]
        # Also consider items with unresolved refs that ARE in evidence
        strong_with_resolved = [
            item for item in relevant_items
            if item.relevance_score >= cfg.strong_item_score
        ]

        # Additional check: if the most topically relevant evidence item
        # (highest relevance_score, computed from our signal analysis)
        # has unresolved cross-references AND delegates to them,
        # the primary evidence for this question is incomplete.
        # We use relevance_score (not retrieval combined_score) because
        # relevance_score reflects actual content alignment with the query,
        # not retrieval rank which can be dominated by BM25 coincidences.
        top_by_relevance = evidence_items[0]  # sorted by relevance desc
        top_delegates_unresolved = (
            bool(top_by_relevance.unresolved_cross_refs)
            and "delegates_to_cross_reference" in top_by_relevance.gap_signals
            and top_by_relevance.relevance_score >= cfg.strong_item_score
        )

        if (
            support_score >= cfg.min_support
            and strong_with_resolved
            and not _all_items_delegate(relevant_items)
            and not top_delegates_unresolved
        ):
            primary = tuple(i.clause for i in strong_with_resolved[:3])
            return _make_decision(
                status=DecisionStatus.SUPPORTED,
                question=question,
                evidence=tuple(evidence_items),
                rationale=(
                    f"Evidence support score {support_score:.3f} >= "
                    f"threshold {cfg.min_support:.3f}. "
                    f"{len(strong_with_resolved)} strong evidence item(s) found."
                ),
                support_score=support_score,
                primary_clauses=primary,
                conflict_details=(),
                missing_information="",
                recommended_action="generate_answer",
            )

        # 6d. Default: INSUFFICIENT
        top_item = relevant_items[0] if relevant_items else evidence_items[0]
        gap_reasons = _summarise_gaps(relevant_items)
        delegation_note = (
            f" Most relevant clause (§{top_by_relevance.clause.clause_id}) "
            f"delegates to unresolved references: "
            f"{', '.join(top_by_relevance.unresolved_cross_refs)}."
            if top_delegates_unresolved else ""
        )
        return _make_decision(
            status=DecisionStatus.INSUFFICIENT,
            question=question,
            evidence=tuple(evidence_items),
            rationale=(
                f"Evidence support score {support_score:.3f} below "
                f"threshold {cfg.min_support:.3f}, or no strong item "
                f"found (best: {top_item.relevance_score:.3f}). "
                + (f"Gap signals: {gap_reasons}." if gap_reasons else "")
                + delegation_note
            ),
            support_score=support_score,
            primary_clauses=tuple(i.clause for i in relevant_items[:3]),
            conflict_details=(),
            missing_information=gap_reasons or delegation_note.strip() or (
                "The retrieved evidence does not sufficiently settle the question."
            ),
            recommended_action="refuse",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "is", "are", "was", "were",
    "to", "for", "by", "on", "at", "as", "any", "that", "this",
    "with", "or", "and", "not", "be", "may", "must", "shall",
    "will", "has", "have", "had", "been", "such", "where", "which",
    "from", "under", "its", "their", "all", "each", "than", "then",
    "so", "but", "if", "do", "did", "no", "up", "out", "how",
    "what", "when", "who", "does", "i", "my",
})


def _meaningful_query_tokens(question: str) -> frozenset[str]:
    words = re.findall(r'\b[a-z0-9]{2,}\b', question.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS)


def _ref_id(ref: str) -> str:
    """Strip § prefix from a cross-reference."""
    return ref.lstrip("§")


def _unique_clauses_from_conflicts(
    conflicts: list[ConflictDetail],
) -> list:
    seen: set[str] = set()
    clauses = []
    for c in conflicts:
        for clause in (c.clause_a, c.clause_b):
            if clause.clause_id not in seen:
                seen.add(clause.clause_id)
                clauses.append(clause)
    return clauses


def _all_items_delegate(items: list[EvidenceItem]) -> bool:
    """Return True if ALL relevant items simply delegate to a cross-reference."""
    if not items:
        return False
    return all("delegates_to_cross_reference" in item.gap_signals for item in items)


def _summarise_gaps(items: list[EvidenceItem]) -> str:
    all_gaps: list[str] = []
    for item in items:
        for g in item.gap_signals:
            if g not in all_gaps:
                all_gaps.append(g)
    return "; ".join(all_gaps[:3]) if all_gaps else ""


def _make_decision(
    *,
    status: DecisionStatus,
    question: str,
    evidence: tuple[EvidenceItem, ...],
    rationale: str,
    support_score: float,
    primary_clauses: tuple,
    conflict_details: tuple,
    missing_information: str,
    recommended_action: str,
) -> EvidenceDecision:
    return EvidenceDecision(
        status=status,
        question=question,
        evidence=evidence,
        rationale=rationale,
        support_score=support_score,
        primary_clauses=primary_clauses,
        conflict_details=conflict_details,
        missing_information=missing_information,
        recommended_action=recommended_action,
    )
