"""
Hybrid retrieval: combines semantic (vector) and lexical (BM25) retrieval.

Merging strategy — Reciprocal Rank Fusion (RRF)
------------------------------------------------
RRF is a simple, parameter-light fusion method that does not require
score normalisation across the two retrieval systems.

Formula (Cormack et al., 2009):

    RRF(clause) = Σ  1 / (k + rank_i)

where ``rank_i`` is the 1-based rank of the clause in system i, and
``k`` is a smoothing constant (default 60 — the standard value from the
original paper).

Why RRF:
- Semantic cosine scores and BM25 scores are on different scales and
  cannot be added directly without calibration.
- RRF depends only on rank position, not raw score magnitude, making it
  robust to scale differences.
- It has a well-documented empirical record in information retrieval.
- It is trivial to implement correctly with no additional dependencies.

Clauses that appear in only one result set receive a contribution from
that system only.  Clauses that appear in both receive contributions from
both, naturally boosting overlap candidates.

The individual ``semantic_score`` and ``lexical_score`` fields on
``RetrievalResult`` are preserved for transparency and potential
downstream analysis.  The ``combined_score`` is the RRF score.

Configuration
-------------
All parameters have sensible defaults but are fully configurable:

    semantic_top_k   — candidates from the semantic index   (default 10)
    lexical_top_k    — candidates from the lexical index    (default 10)
    final_top_k      — results returned to the caller       (default 10)
    rrf_k            — RRF smoothing constant               (default 60)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.parser import PolicyClause
from src.retrieval.embeddings import EmbeddingProvider
from src.retrieval.lexical import LexicalIndex
from src.retrieval.models import RetrievalResult
from src.retrieval.vector import VectorIndex


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrieverConfig:
    """
    Configuration for the ``HybridRetriever``.

    All fields have sensible defaults suitable for the 137-clause corpus.
    """
    semantic_top_k: int = 10
    lexical_top_k: int = 10
    final_top_k: int = 10
    rrf_k: int = 60


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Retrieve the most relevant policy clauses for a query using both
    semantic and lexical methods, merged by Reciprocal Rank Fusion.

    Build from a populated ``ClauseStore`` (or any list of clauses):

    ::

        from src.ingestion import load_policy_document, parse_clauses, ClauseStore
        from src.retrieval import HybridRetriever, RetrieverConfig
        from src.retrieval.embeddings import OpenAIEmbeddingProvider

        doc = load_policy_document("data/raw/policy_manual.md")
        store = ClauseStore(parse_clauses(doc))

        provider = OpenAIEmbeddingProvider()
        retriever = HybridRetriever.build(store.all(), provider)

        results = retriever.retrieve("How many days to report a change?", top_k=5)
        for r in results:
            print(r.clause.clause_id, r.combined_score)

    Parameters
    ----------
    vector_index:
        Pre-built ``VectorIndex``.
    lexical_index:
        Pre-built ``LexicalIndex``.
    config:
        Retrieval parameters.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._vector = vector_index
        self._lexical = lexical_index
        self._config = config or RetrieverConfig()

    @classmethod
    def build(
        cls,
        clauses: list[PolicyClause],
        provider: EmbeddingProvider,
        config: RetrieverConfig | None = None,
    ) -> "HybridRetriever":
        """
        Build both indexes from *clauses* and return a ready retriever.

        Parameters
        ----------
        clauses:
            All clauses to index (e.g. ``store.all()``).
        provider:
            Embedding provider.  Use ``OpenAIEmbeddingProvider`` in
            production and ``FakeEmbeddingProvider`` in tests.
        config:
            Optional retrieval configuration.
        """
        cfg = config or RetrieverConfig()
        vector_index = VectorIndex.build(clauses, provider)
        lexical_index = LexicalIndex.build(clauses)
        return cls(vector_index, lexical_index, cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant policy clauses for *query*.

        Parameters
        ----------
        query:
            Natural-language question or keyword query.
        top_k:
            Maximum number of results to return.  Defaults to
            ``config.final_top_k`` if not specified.

        Returns
        -------
        list[RetrievalResult]
            Sorted by descending ``combined_score``.
            Each result retains the full ``PolicyClause`` record.
        """
        cfg = self._config
        k_final = top_k if top_k is not None else cfg.final_top_k

        if not query.strip():
            return []

        # --- Retrieve candidates from each index ---
        sem_results = self._vector.search(query, top_k=cfg.semantic_top_k)
        lex_results = self._lexical.search(query, top_k=cfg.lexical_top_k)

        # --- Build per-clause score dictionaries ---
        # Key: clause_id  Value: (clause, sem_score, lex_score)
        by_id: dict[str, _CandidateAccumulator] = {}

        for rank, (clause, score) in enumerate(sem_results, start=1):
            cid = clause.clause_id
            if cid not in by_id:
                by_id[cid] = _CandidateAccumulator(clause)
            by_id[cid].add_semantic(score, rank, cfg.rrf_k)

        for rank, (clause, score) in enumerate(lex_results, start=1):
            cid = clause.clause_id
            if cid not in by_id:
                by_id[cid] = _CandidateAccumulator(clause)
            by_id[cid].add_lexical(score, rank, cfg.rrf_k)

        # --- Build RetrievalResult objects and rank ---
        results = [acc.to_result() for acc in by_id.values()]
        results.sort(key=lambda r: r.combined_score, reverse=True)

        return results[:k_final]


# ---------------------------------------------------------------------------
# Internal accumulator (not part of the public API)
# ---------------------------------------------------------------------------

class _CandidateAccumulator:
    """Collects semantic and lexical contributions for one clause."""

    def __init__(self, clause: PolicyClause) -> None:
        self.clause = clause
        self.semantic_score: float = 0.0
        self.lexical_score: float = 0.0
        self._rrf_total: float = 0.0
        self._sources: list[str] = []

    def add_semantic(self, score: float, rank: int, k: int) -> None:
        self.semantic_score = score
        self._rrf_total += 1.0 / (k + rank)
        self._sources.append("semantic")

    def add_lexical(self, score: float, rank: int, k: int) -> None:
        self.lexical_score = score
        self._rrf_total += 1.0 / (k + rank)
        self._sources.append("lexical")

    def to_result(self) -> RetrievalResult:
        sources_tuple: tuple = tuple(sorted(set(self._sources)))  # type: ignore[assignment]
        return RetrievalResult(
            clause=self.clause,
            semantic_score=self.semantic_score,
            lexical_score=self.lexical_score,
            combined_score=self._rrf_total,
            sources=sources_tuple,
        )
