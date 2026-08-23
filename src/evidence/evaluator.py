"""
Evidence evaluator — core decision logic.

Decision flow
-------------
1. Receive the question and retrieved evidence (list[RetrievalResult]).
2. Build EvidenceItem objects: compute relevance scores and extract
   support/gap signals for each retrieved clause.
3. Detect conflicts between evidence items (contradiction check).
4. Apply decision rules:
   a. If key query concepts are missing from all evidence: INSUFFICIENT.
   b. If no relevant evidence: INSUFFICIENT.
   c. If material conflict detected: CONFLICTING.
   d. If all topic-specific clauses delegate or have unresolved gaps: INSUFFICIENT.
   e. If aggregate support score >= threshold AND at least one complete strong
      evidence item with substantive topic overlap: SUPPORTED.
   f. Otherwise: INSUFFICIENT (conservative default).
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
from src.ingestion.parser import PolicyClause
from src.retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvidenceConfig:
    """
    Thresholds and parameters for the evidence evaluator.
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
                if not _is_ref_topically_resolved(
                    _ref_id(ref), retrieved_ids, deduped_results, query_tokens
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

        # --- Step 4: Detect missing external query concepts ---
        missing_concepts = _find_missing_query_concepts(
            question, [item.clause for item in evidence_items[:5]]
        )
        if missing_concepts and len(missing_concepts) >= 2:
            return _make_decision(
                status=DecisionStatus.INSUFFICIENT,
                question=question,
                evidence=tuple(evidence_items),
                rationale=(
                    f"The policy manual does not contain provisions addressing: "
                    f"{', '.join(missing_concepts)}."
                ),
                support_score=0.0,
                primary_clauses=tuple(i.clause for i in evidence_items[:3]),
                conflict_details=(),
                missing_information=(
                    f"No policy provisions mention {', '.join(missing_concepts)}."
                ),
                recommended_action="refuse",
            )

        # --- Step 5: Detect conflicts ---
        conflicts = detect_conflicts(
            evidence_items,
            conflict_min_relevance=cfg.conflict_min_relevance,
        )

        # --- Step 6: Aggregate support score ---
        if relevant_items:
            support_score = sum(
                item.relevance_score for item in relevant_items
            ) / len(relevant_items)
        else:
            support_score = 0.0

        # --- Step 7: Apply decision rules ---

        # 7a. No relevant evidence above threshold
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

        # 7b. Material conflict
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

        # 7c. Check if topic-specific items all delegate or have gaps
        topic_items = [
            item for item in relevant_items
            if any(s.startswith("lexical_overlap") for s in item.support_signals)
        ]
        topic_items_all_delegate = (
            bool(topic_items)
            and all(
                ("delegates_to_cross_reference" in item.gap_signals or bool(item.unresolved_cross_refs))
                for item in topic_items
            )
        )

        # 7d. Complete strong items that directly settle the rule without delegation
        complete_strong_items = [
            item for item in relevant_items
            if item.relevance_score >= cfg.strong_item_score
            and not item.unresolved_cross_refs
            and "delegates_to_cross_reference" not in item.gap_signals
            and any(s.startswith("lexical_overlap") for s in item.support_signals)
        ]

        if (
            support_score >= cfg.min_support
            and complete_strong_items
            and not _all_items_delegate(relevant_items)
            and not topic_items_all_delegate
        ):
            primary = tuple(i.clause for i in complete_strong_items[:3])
            return _make_decision(
                status=DecisionStatus.SUPPORTED,
                question=question,
                evidence=tuple(evidence_items),
                rationale=(
                    f"Evidence support score {support_score:.3f} >= "
                    f"threshold {cfg.min_support:.3f}. "
                    f"{len(complete_strong_items)} complete strong evidence item(s) found."
                ),
                support_score=support_score,
                primary_clauses=primary,
                conflict_details=(),
                missing_information="",
                recommended_action="generate_answer",
            )

        # 7e. Default: INSUFFICIENT
        top_item = relevant_items[0] if relevant_items else evidence_items[0]
        gap_reasons = _summarise_gaps(relevant_items)
        has_delegations = any(
            "delegates_to_cross_reference" in i.gap_signals or bool(i.unresolved_cross_refs)
            for i in (topic_items or relevant_items)
        )
        delegation_note = (
            f" Relevant clause (§{top_item.clause.clause_id}) "
            f"delegates or has unresolved references."
            if has_delegations else ""
        )
        return _make_decision(
            status=DecisionStatus.INSUFFICIENT,
            question=question,
            evidence=tuple(evidence_items),
            rationale=(
                f"Evidence support score {support_score:.3f} below "
                f"threshold {cfg.min_support:.3f}, or no complete substantive clause "
                f"found (best: {top_item.relevance_score:.3f}). "
                + (f"Gap signals: {gap_reasons}." if gap_reasons else "")
                + delegation_note
            ),
            support_score=support_score,
            primary_clauses=tuple(i.clause for i in (topic_items or relevant_items)[:3]),
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
    "what", "when", "who", "does", "i", "my", "can", "an",
    "policy", "policies", "rule", "rules", "time", "times",
    "information", "detail", "details", "tell", "about",
})


def _meaningful_query_tokens(question: str) -> frozenset[str]:
    words = re.findall(r'\b[a-z0-9]{2,}\b', question.lower())
    base = set(w for w in words if w not in _STOP_WORDS)
    for w in list(base):
        if len(w) > 3:
            if w.endswith('s') and not w.endswith('ss'):
                base.add(w[:-1])
            else:
                base.add(w + 's')
    return frozenset(base)


def _find_missing_query_concepts(
    question: str,
    retrieved_clauses: list[PolicyClause],
) -> list[str]:
    """Identify key entities/nouns in the question that do not appear anywhere in retrieved evidence."""
    domain_common = {
        "what", "is", "are", "the", "for", "can", "an", "applicant", "applicants",
        "recipient", "recipients", "household", "households", "policy", "how",
        "many", "does", "have", "has", "had", "to", "a", "of", "in", "and", "or",
        "directly", "any", "must", "should", "could", "would", "when", "where",
        "which", "who", "whom", "whose", "why", "under", "with", "from", "into",
        "about", "their", "they", "them", "there", "then", "than", "this", "that",
        "these", "those", "county", "program", "department", "support", "member",
        "members", "receive", "receiving", "provide", "provided", "information",
        "rules", "rule", "make", "made", "been", "being", "allow", "allowed", "tell",
        "limit", "limits", "income", "resource", "resources", "date", "dates", "time",
        "times", "month", "months", "monthly", "year", "years", "annual", "annually",
        "amount", "amounts", "change", "changes", "report", "reporting", "reported",
        "policy", "policies",
    }
    q_tokens = [
        w for w in re.findall(r'\b[a-z]{4,}\b', question.lower())
        if w not in domain_common
    ]
    if not q_tokens:
        return []

    combined_text = " ".join(c.text.lower() for c in retrieved_clauses)
    missing = []
    for w in q_tokens:
        stem = w[:-1] if w.endswith('s') and not w.endswith('ss') else w
        if stem not in combined_text and w not in combined_text:
            missing.append(w)
    return missing


def _ref_id(ref: str) -> str:
    """Strip § prefix from a cross-reference."""
    return ref.lstrip("§")


def _is_ref_topically_resolved(
    ref_id: str,
    retrieved_ids: frozenset[str],
    results: list[RetrievalResult],
    query_tokens: frozenset[str],
) -> bool:
    """
    Return True if a cross-reference is genuinely resolved by the retrieved evidence.
    """
    if ref_id in retrieved_ids:
        return True

    for result in results:
        cid = result.clause.clause_id
        if cid.startswith(ref_id + "."):
            clause_tokens = frozenset(
                re.findall(r'\b[a-z]{3,}\b', result.clause.text.lower())
            )
            meaningful_query = {
                t for t in query_tokens
                if t not in _STOP_WORDS and len(t) >= 3
            }
            overlap = meaningful_query & clause_tokens
            if len(overlap) >= 2:
                return True

    return False


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
