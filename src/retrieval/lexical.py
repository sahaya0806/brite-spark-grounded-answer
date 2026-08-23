"""
BM25 lexical index for policy clause retrieval.

Design notes
------------
- BM25Okapi from the ``rank-bm25`` library is used.  It is a standard,
  well-tested BM25 implementation with no additional dependencies.
- Tokenisation preserves meaningful policy terms: numbers, percentages,
  monetary values, durations, and clause identifiers (e.g. "4.3.2") are
  kept intact.  We do not use NLTK stopword removal because policy
  vocabulary — "must", "may", "not", "no" — carries legal meaning.
- Scores returned by BM25Okapi are non-negative but unbounded.  We
  normalise them to [0, 1] by dividing by the maximum score in each
  result set.  If all scores are 0, normalised scores are 0.
- The index operates over a combined text per clause that includes:
    - clause ID (§4.3.2 and 4.3.2 both tokenise to useful terms)
    - section title
    - clause body text
    - sub-item texts
  This ensures that a query mentioning a clause number, a section topic,
  or exact policy wording all have good lexical coverage.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.ingestion.parser import PolicyClause


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Keep alphanumeric runs and decimal numbers (e.g. "4.3.2", "$1,500", "90%")
# as single tokens.  Split on everything else.
_TOKEN_RE = re.compile(r"[\w.,$%]+")


def tokenise(text: str) -> list[str]:
    """
    Split *text* into lowercase tokens, preserving policy-relevant terms.

    Numbers, clause IDs, monetary amounts, and percentages are kept as
    single tokens.  Empty token lists are returned for empty text.
    """
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class LexicalIndex:
    """
    BM25Okapi index over the clause corpus.

    Usage
    -----
    ::

        index = LexicalIndex.build(clauses)
        results = index.search("10 calendar days report change", top_k=5)
        # results: list of (PolicyClause, normalised_score)

    Parameters
    ----------
    clauses:
        All ``PolicyClause`` objects to index, in document order.
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        clauses: list[PolicyClause],
    ) -> None:
        self._bm25 = bm25
        self._clauses = clauses

    @classmethod
    def build(cls, clauses: list[PolicyClause]) -> "LexicalIndex":
        """
        Build the BM25 index from *clauses*.

        Parameters
        ----------
        clauses:
            Non-empty list of ``PolicyClause`` objects.
        """
        if not clauses:
            raise ValueError("Cannot build a LexicalIndex with no clauses.")

        tokenised_corpus = [
            tokenise(_clause_searchable_text(c)) for c in clauses
        ]
        bm25 = BM25Okapi(tokenised_corpus)
        return cls(bm25, list(clauses))

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[PolicyClause, float]]:
        """
        Retrieve the top-k clauses by BM25 score for *query*.

        Parameters
        ----------
        query:
            Natural-language or keyword query.
        top_k:
            Maximum number of clauses to return.

        Returns
        -------
        list of (PolicyClause, normalised_score)
            Sorted by descending BM25 score.  Scores are normalised to
            [0, 1] within this result set.  Clauses with score 0 are
            excluded.
        """
        if not query.strip():
            return []

        tokens = tokenise(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Pair with clauses, filter zero-score, normalise, sort, truncate
        pairs = list(zip(raw_scores, self._clauses))
        max_score = max(s for s, _ in pairs) if pairs else 0.0

        results: list[tuple[PolicyClause, float]] = []
        for raw, clause in pairs:
            if raw <= 0.0:
                continue
            norm = float(raw / max_score) if max_score > 0.0 else 0.0
            results.append((clause, norm))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def _clause_searchable_text(clause: PolicyClause) -> str:
    """
    Assemble the full searchable text for a clause.

    Includes:
    - The clause ID in §-prefixed and bare numeric forms.
    - The section title (part context).
    - The full clause body text.
    - All sub-item texts (already embedded in clause.text but included
      explicitly to ensure they receive full BM25 weight).
    """
    parts = [
        f"§{clause.clause_id}",
        clause.clause_id,
        clause.section_title,
        clause.text,
    ]
    for item in clause.sub_items:
        parts.append(item.text)
    return " ".join(p for p in parts if p)
