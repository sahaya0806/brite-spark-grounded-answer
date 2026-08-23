"""
Evidence scoring and signal extraction.

This module computes per-clause relevance scores and extracts explicit
support/gap signals from evidence clauses against the user's question.

Design principles
-----------------
- All scoring is deterministic (no randomness, no LLM calls).
- Scores combine retrieval quality with lexical content signals.
- Support signals describe WHY a clause is considered relevant.
- Gap signals describe WHY a clause may be insufficient.
- Numeric/temporal values are explicitly extracted for contradiction detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.parser import PolicyClause
from src.retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Patterns for policy content analysis
# ---------------------------------------------------------------------------

# Duration / deadline values: "10 days", "30 calendar days", "28 days", etc.
_RE_DURATION = re.compile(
    r'\b(\d+)\s*(calendar\s+)?days?\b', re.IGNORECASE
)

# Monetary values: "$4,000", "$120 per month", etc.
_RE_MONETARY = re.compile(
    r'\$[\d,]+(?:\s+per\s+\w+)?', re.IGNORECASE
)

# Percentage values: "10 per cent", "20%"
_RE_PERCENT = re.compile(
    r'\b(\d+)\s*(?:per\s+cent|%)\b', re.IGNORECASE
)

# Obligation language
_RE_OBLIGATION = re.compile(
    r'\b(must|shall|required|obliged|obligated)\b', re.IGNORECASE
)

# Eligibility / condition language
_RE_ELIGIBILITY = re.compile(
    r'\b(eligible|eligib|qualify|qualif|entitled|entitlement|condition|'
    r'satisf|meet|meets|does not|must not|shall not)\b',
    re.IGNORECASE,
)

# Reporting / notification language
_RE_REPORTING = re.compile(
    r'\b(report|notify|notif|inform|disclose|declare|chang)\w*\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extracted numeric fact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericFact:
    """A numeric or temporal value extracted from a clause."""
    kind: str    # "duration_days", "monetary", "percentage"
    raw: str     # the matched text as it appears in the clause
    value: str   # normalised string value for comparison


def extract_numeric_facts(text: str) -> tuple[NumericFact, ...]:
    """Extract numeric/temporal facts from *text*."""
    facts: list[NumericFact] = []

    for m in _RE_DURATION.finditer(text):
        facts.append(NumericFact(
            kind="duration_days",
            raw=m.group(0),
            value=m.group(1),  # just the number
        ))

    for m in _RE_MONETARY.finditer(text):
        facts.append(NumericFact(
            kind="monetary",
            raw=m.group(0),
            value=m.group(0),
        ))

    for m in _RE_PERCENT.finditer(text):
        facts.append(NumericFact(
            kind="percentage",
            raw=m.group(0),
            value=m.group(1),
        ))

    return tuple(facts)


# ---------------------------------------------------------------------------
# Support / gap signals
# ---------------------------------------------------------------------------

def extract_support_signals(
    clause: PolicyClause,
    query_tokens: frozenset[str],
) -> tuple[str, ...]:
    """
    Identify textual signals that suggest *clause* supports the query.

    Parameters
    ----------
    clause:
        The policy clause to inspect.
    query_tokens:
        Lowercased meaningful tokens from the user's question.

    Returns
    -------
    Tuple of signal description strings.
    """
    signals: list[str] = []
    text = clause.text.lower()

    # Obligation language present
    if _RE_OBLIGATION.search(text):
        signals.append("contains_obligation_language")

    # Eligibility / condition language present
    if _RE_ELIGIBILITY.search(text):
        signals.append("contains_eligibility_language")

    # Reporting language present (relevant for deadline questions)
    if _RE_REPORTING.search(text):
        signals.append("contains_reporting_language")

    # Numeric/temporal facts present
    facts = extract_numeric_facts(clause.text)
    if any(f.kind == "duration_days" for f in facts):
        signals.append("contains_duration_value")
    if any(f.kind == "monetary" for f in facts):
        signals.append("contains_monetary_value")

    # Query term overlap with clause text (content relevance)
    clause_tokens = frozenset(re.findall(r'\w+', text))
    overlap = query_tokens & clause_tokens
    important_overlap = {
        t for t in overlap
        if len(t) > 3  # skip very short words
    }
    if len(important_overlap) >= 3:
        signals.append(f"lexical_overlap:{','.join(sorted(important_overlap)[:5])}")

    # Sub-items present (richer clause, more likely to be the primary rule)
    if clause.sub_items:
        signals.append(f"has_{len(clause.sub_items)}_sub_items")

    return tuple(signals)


def extract_gap_signals(
    clause: PolicyClause,
    retrieved_ids: frozenset[str],
    query_tokens: frozenset[str],
) -> tuple[str, ...]:
    """
    Identify signals that suggest *clause* may be insufficient alone.

    Parameters
    ----------
    clause:
        The policy clause to inspect.
    retrieved_ids:
        Clause IDs of all retrieved evidence.
    query_tokens:
        Lowercased meaningful tokens from the user's question.

    Returns
    -------
    Tuple of gap signal description strings.
    """
    signals: list[str] = []

    # Unresolved cross-references
    unresolved = _unresolved_refs(clause, retrieved_ids)
    if unresolved:
        signals.append(
            f"unresolved_cross_refs:{','.join(unresolved)}"
        )

    # Clause refers elsewhere for the main substance (incomplete answer)
    ref_verbs = re.findall(
        r'\b(see|refer|addressed|covered|under|within)\b',
        clause.text, re.IGNORECASE,
    )
    if ref_verbs and clause.cross_references:
        signals.append("delegates_to_cross_reference")

    return tuple(signals)


def _unresolved_refs(
    clause: PolicyClause,
    retrieved_ids: frozenset[str],
) -> list[str]:
    """Return cross-references in *clause* that are not in *retrieved_ids*."""
    unresolved = []
    for ref in clause.cross_references:
        # ref is "§N.N.N" — strip the § for comparison
        ref_id = ref.lstrip("§")
        # Also match two-component refs (e.g. "§6.4" matches any "6.4.x")
        if ref_id not in retrieved_ids:
            if not any(r.startswith(ref_id + ".") for r in retrieved_ids):
                unresolved.append(ref)
    return unresolved


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def compute_relevance_score(
    result: RetrievalResult,
    support_signals: tuple[str, ...],
) -> float:
    """
    Compute a relevance score in [0, 1] for one evidence item.

    The score combines retrieval quality with the presence of positive
    content signals.  It does NOT represent answer confidence.

    Formula:
        base  = weighted_average(semantic_score, lexical_score)
        bonus = min(0.15, 0.05 * n_support_signals)
        score = min(1.0, base + bonus)

    The bonus rewards clauses that contain policy-substantive content
    matching the question, but is capped to prevent inflation.
    """
    sem = result.semantic_score
    lex = result.lexical_score

    # Weight: if both available, weight lexical slightly more for policy text
    if sem > 0.0 and lex > 0.0:
        base = 0.45 * sem + 0.55 * lex
    elif sem > 0.0:
        base = sem * 0.85  # slight penalty for semantic-only
    elif lex > 0.0:
        base = lex * 0.90
    else:
        base = 0.0

    # Content signal bonus
    content_signals = [
        s for s in support_signals
        if not s.startswith("lexical_overlap")
    ]
    bonus = min(0.15, 0.04 * len(content_signals))
    return min(1.0, base + bonus)
