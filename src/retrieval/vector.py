"""
FAISS-backed vector index for semantic clause retrieval.

Design notes
------------
- The corpus is 137 clauses.  A flat exact-search FAISS index
  (``IndexFlatIP``) is entirely appropriate at this scale.  Approximate
  nearest-neighbour indexes (IVF, HNSW) would add complexity with no
  measurable benefit for 137 vectors.
- Vectors are L2-normalised before insertion so that inner-product search
  (``IndexFlatIP``) is equivalent to cosine similarity.
- The index preserves the mapping: FAISS integer position → PolicyClause,
  so retrieval always returns structured clause records.
- Returned semantic scores are in [0, 1] (cosine similarity after
  normalisation, clipped to avoid floating-point noise).
"""

from __future__ import annotations

import numpy as np
import faiss
from numpy.typing import NDArray

from src.ingestion.parser import PolicyClause
from src.retrieval.embeddings import EmbeddingProvider


def _normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalise a batch of vectors row-wise (in-place safe copy)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class VectorIndex:
    """
    FAISS flat inner-product index over policy clause embeddings.

    Usage
    -----
    ::

        index = VectorIndex.build(clauses, provider)
        results = index.search(query_text, top_k=5)
        # results: list of (PolicyClause, semantic_score)

    Parameters
    ----------
    clauses:
        List of ``PolicyClause`` objects in the order they were indexed.
    provider:
        An ``EmbeddingProvider`` used both to build and to query the index.
    """

    def __init__(
        self,
        faiss_index: faiss.IndexFlatIP,
        clauses: list[PolicyClause],
        provider: EmbeddingProvider,
    ) -> None:
        self._index = faiss_index
        self._clauses = clauses
        self._provider = provider

    @classmethod
    def build(
        cls,
        clauses: list[PolicyClause],
        provider: EmbeddingProvider,
    ) -> "VectorIndex":
        """
        Embed all clauses and build the FAISS index.

        Parameters
        ----------
        clauses:
            All clauses to index.  Must not be empty.
        provider:
            Embedding provider used to generate clause vectors.

        Returns
        -------
        VectorIndex
        """
        if not clauses:
            raise ValueError("Cannot build a VectorIndex with no clauses.")

        texts = [_clause_text(c) for c in clauses]
        raw = provider.embed_documents(texts)
        vectors = _normalise(raw)

        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)

        return cls(faiss_index, list(clauses), provider)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[PolicyClause, float]]:
        """
        Retrieve the top-k clauses most semantically similar to *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of clauses to return.

        Returns
        -------
        list of (PolicyClause, score)
            Sorted by descending cosine similarity.  Scores are in [0, 1].
        """
        if not query.strip():
            return []

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        q_raw = self._provider.embed_query(query)
        q_vec = _normalise(q_raw.reshape(1, -1))

        scores, indices = self._index.search(q_vec, k)

        results: list[tuple[PolicyClause, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            # Clip to [0, 1] — floating-point arithmetic can produce
            # values very slightly above 1.0 for identical vectors.
            clipped = float(np.clip(score, 0.0, 1.0))
            results.append((self._clauses[idx], clipped))

        return results


def _clause_text(clause: PolicyClause) -> str:
    """
    Assemble the searchable text representation of a clause.

    Includes the clause ID, section title (provides topic context),
    and the full clause text (which already includes sub-items and tables).
    """
    parts = [
        f"§{clause.clause_id}",
        clause.section_title,
        clause.text,
    ]
    return " ".join(p for p in parts if p)
