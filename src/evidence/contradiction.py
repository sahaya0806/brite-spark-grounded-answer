"""
Contradiction detection between policy clause evidence items.

Design principle
----------------
Contradictions are detected by examining pairs of evidence clauses for
competing values on the same kind of fact (durations, monetary figures,
percentages) within closely related policy context.

We do NOT hard-code specific clause IDs.  The detection is general:
if two clauses retrieved for the same question both contain "N days"
obligations where N differs, that is a contradiction signal.

The known corpus case — §4.3.2 (10 days) vs §9.1.4 (30 days) — is
detected by this general mechanism, not by a special-case rule.

Conservative policy
-------------------
A conflict is only raised when:
1. Both clauses contain numeric facts of the same kind (e.g. both
   contain duration_days facts).
2. The numeric values differ.
3. Both clauses have meaningful relevance to the question
   (relevance_score >= conflict_min_relevance threshold).

This avoids false positives from clauses that mention different numbers
in genuinely different contexts (e.g. "28 days" for absence vs "30 days"
for review — these are different subjects).

To distinguish same-subject from different-subject, we additionally check
for lexical overlap between the two clauses.  Clauses that share multiple
significant terms are more likely to address the same policy question.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.evidence.models import ConflictDetail, EvidenceItem
from src.evidence.scoring import NumericFact, extract_numeric_facts


# Minimum shared token count to consider two clauses as covering the
# same policy subject.
_MIN_SHARED_TOKENS = 3

# Short words to exclude from overlap check.
_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "is", "are", "was", "were",
    "to", "for", "by", "on", "at", "as", "any", "that", "this",
    "with", "or", "and", "not", "be", "may", "must", "shall",
    "will", "has", "have", "had", "been", "such", "where", "which",
    "from", "under", "its", "their", "all", "each", "than", "then",
    "so", "but", "if", "do", "did", "no", "up", "out",
})


def _meaningful_tokens(text: str) -> frozenset[str]:
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS)


def _facts_by_kind(facts: tuple[NumericFact, ...]) -> dict[str, list[NumericFact]]:
    out: dict[str, list[NumericFact]] = {}
    for f in facts:
        out.setdefault(f.kind, []).append(f)
    return out


def detect_conflicts(
    evidence_items: Sequence[EvidenceItem],
    conflict_min_relevance: float = 0.10,
) -> list[ConflictDetail]:
    """
    Detect contradictions between pairs of evidence items.

    Parameters
    ----------
    evidence_items:
        All evidence items considered for the question.
    conflict_min_relevance:
        Minimum relevance score for a clause to participate in conflict
        detection.  Items below this threshold are ignored.

    Returns
    -------
    list[ConflictDetail]
        One entry per detected conflict pair.  May be empty.
    """
    # Filter to items with enough relevance to matter
    relevant = [
        item for item in evidence_items
        if item.relevance_score >= conflict_min_relevance
    ]

    if len(relevant) < 2:
        return []

    conflicts: list[ConflictDetail] = []

    for i in range(len(relevant)):
        for j in range(i + 1, len(relevant)):
            item_a = relevant[i]
            item_b = relevant[j]

            # Skip clauses from the same section (same section = probably
            # complementary, not contradictory)
            if item_a.clause.section_id == item_b.clause.section_id:
                continue

            conflict = _check_pair(item_a, item_b)
            if conflict is not None:
                conflicts.append(conflict)

    return conflicts


def _check_pair(
    item_a: EvidenceItem,
    item_b: EvidenceItem,
) -> ConflictDetail | None:
    """
    Check a single pair of evidence items for a conflict.

    Returns a ConflictDetail if a conflict is detected, else None.
    """
    clause_a = item_a.clause
    clause_b = item_b.clause

    # Check whether the two clauses share enough vocabulary to be
    # addressing the same policy subject.
    tokens_a = _meaningful_tokens(clause_a.text)
    tokens_b = _meaningful_tokens(clause_b.text)
    shared = tokens_a & tokens_b
    if len(shared) < _MIN_SHARED_TOKENS:
        return None

    # Extract numeric facts from each clause
    facts_a = _facts_by_kind(extract_numeric_facts(clause_a.text))
    facts_b = _facts_by_kind(extract_numeric_facts(clause_b.text))

    # Check each fact kind that appears in both clauses
    for kind in facts_a:
        if kind not in facts_b:
            continue

        values_a = {f.value for f in facts_a[kind]}
        values_b = {f.value for f in facts_b[kind]}

        # If both clauses mention the same kind of numeric fact but
        # with DIFFERENT values, that is a potential conflict.
        differing = values_a.symmetric_difference(values_b)
        if not differing:
            continue  # same values — no conflict

        # Format the specific conflicting values for the report
        raw_a = ", ".join(f.raw for f in facts_a[kind])
        raw_b = ", ".join(f.raw for f in facts_b[kind])

        kind_label = _kind_label(kind)
        explanation = (
            f"§{clause_a.clause_id} states {raw_a!r} while "
            f"§{clause_b.clause_id} states {raw_b!r} for the same "
            f"{kind_label} obligation. "
            f"Both clauses share policy vocabulary: "
            f"{', '.join(sorted(shared)[:6])}."
        )

        return ConflictDetail(
            clause_a=clause_a,
            clause_b=clause_b,
            conflict_type=f"competing_{kind}",
            value_a=raw_a,
            value_b=raw_b,
            explanation=explanation,
        )

    return None


def _kind_label(kind: str) -> str:
    labels = {
        "duration_days": "day-count",
        "monetary": "monetary",
        "percentage": "percentage",
    }
    return labels.get(kind, kind)
