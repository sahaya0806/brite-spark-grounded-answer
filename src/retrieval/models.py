"""
Retrieval result model.

A RetrievalResult wraps one PolicyClause together with its retrieval
metadata — individual semantic and lexical scores, a combined rank score,
and which retrieval method(s) surfaced this clause.

This model is the contract between the retrieval layer and the evidence
evaluation layer.  Evidence evaluation consumes RetrievalResult objects
and decides whether the evidence supports an answer.

IMPORTANT: This model must NOT contain generated text, LLM output, or any
interpretation of whether the clause answers the question.  It is raw
retrieval evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.ingestion.parser import PolicyClause


# Which retrieval methods contributed a candidate.
RetrievalSource = Literal["semantic", "lexical"]


@dataclass(frozen=True)
class RetrievalResult:
    """
    One policy clause retrieved in response to a query, with retrieval metadata.

    Attributes
    ----------
    clause:
        The full structured ``PolicyClause`` record from the clause store.
        All original fields (clause_id, text, sub_items, cross_references,
        source_path, start_line, end_line, …) are preserved intact.
    semantic_score:
        Cosine similarity in [0, 1] from the semantic (embedding) index.
        0.0 if this clause was not retrieved by the semantic method.
    lexical_score:
        Normalised BM25 score in [0, 1] from the lexical index.
        0.0 if this clause was not retrieved by the lexical method.
    combined_score:
        The merged rank score used to order final results.
        Higher is better.  Computed by the hybrid ranker.
        This score reflects retrieval relevance only.
        It is NOT an answer confidence score.
        It is NOT a refusal threshold.
        It is NOT evidence sufficiency.
    sources:
        Tuple of retrieval method labels that contributed this candidate.
        E.g. ``("semantic",)``, ``("lexical",)``, or
        ``("semantic", "lexical")``.
    """

    clause: PolicyClause
    semantic_score: float
    lexical_score: float
    combined_score: float
    sources: tuple[RetrievalSource, ...]
