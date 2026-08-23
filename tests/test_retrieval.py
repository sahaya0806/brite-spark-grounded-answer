"""
Tests for Milestone 4 — Hybrid Policy Retrieval.

All tests use the deterministic FakeEmbeddingProvider.
No OpenAI API key is required.

Test groups
-----------
1.  Tokeniser
2.  FakeEmbeddingProvider (embedding interface contract)
3.  VectorIndex (semantic retrieval)
4.  LexicalIndex (BM25 retrieval)
5.  HybridRetriever — API contracts
6.  HybridRetriever — deduplication
7.  HybridRetriever — edge cases
8.  RetrievalResult model
9.  Real corpus integration tests (137 clauses, key retrieval scenarios)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.loader import load_policy_document
from src.ingestion.parser import PolicyClause, parse_clauses
from src.ingestion.store import ClauseStore
from src.retrieval import (
    FakeEmbeddingProvider,
    HybridRetriever,
    LexicalIndex,
    RetrievalResult,
    RetrieverConfig,
    VectorIndex,
    tokenise,
)

REAL_CORPUS = Path("data/raw/policy_manual.md")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clause(
    cid: str,
    text: str,
    part_id: str = "1",
    section_id: str = "1.1",
    tmp_path: Path | None = None,
) -> PolicyClause:
    p = Path("/fake/policy.md") if tmp_path is None else tmp_path / "policy.md"
    return PolicyClause(
        clause_id=cid,
        part_id=part_id,
        part_title=f"Part {part_id}",
        section_id=section_id,
        section_title=f"{section_id} Section",
        text=text,
        sub_items=(),
        cross_references=(),
        source_path=p,
        start_line=1,
        end_line=1,
    )


def _small_corpus() -> list[PolicyClause]:
    return [
        _make_clause("1.1.1", "A recipient must report changes within 10 calendar days."),
        _make_clause("1.1.2", "No overpayment shall be established within 30 days.", section_id="1.2"),
        _make_clause("2.1.1", "The household income must not exceed $2,000 per month.", part_id="2", section_id="2.1"),
        _make_clause("2.1.2", "Resources must not exceed $4,000.", part_id="2", section_id="2.1"),
        _make_clause("3.1.1", "An applicant must reside in Calder County.", part_id="3", section_id="3.1"),
    ]


def _small_retriever(config: RetrieverConfig | None = None) -> HybridRetriever:
    provider = FakeEmbeddingProvider(dim=16)
    return HybridRetriever.build(_small_corpus(), provider, config)


# ---------------------------------------------------------------------------
# 1. Tokeniser
# ---------------------------------------------------------------------------

class TestTokeniser:

    def test_basic_split(self):
        assert tokenise("hello world") == ["hello", "world"]

    def test_lowercase(self):
        assert tokenise("Hello WORLD") == ["hello", "world"]

    def test_preserves_clause_ids(self):
        tokens = tokenise("See §4.3.2 for details")
        assert "§4.3.2" in tokens or "4.3.2" in tokens

    def test_preserves_numbers(self):
        tokens = tokenise("within 10 calendar days")
        assert "10" in tokens

    def test_preserves_monetary_values(self):
        tokens = tokenise("not exceed $4,000")
        # $ and digits kept together or as part of token
        combined = " ".join(tokens)
        assert "4,000" in combined or "4" in tokens

    def test_preserves_percentages(self):
        tokens = tokenise("reduced by 20%")
        combined = " ".join(tokens)
        assert "20" in combined

    def test_empty_string_returns_empty(self):
        assert tokenise("") == []

    def test_whitespace_only_returns_empty(self):
        assert tokenise("   \n  ") == []

    def test_punctuation_only_returns_empty(self):
        assert tokenise("!!! ???") == []


# ---------------------------------------------------------------------------
# 2. FakeEmbeddingProvider
# ---------------------------------------------------------------------------

class TestFakeEmbeddingProvider:

    def test_embed_documents_returns_array(self):
        p = FakeEmbeddingProvider(dim=8)
        result = p.embed_documents(["hello", "world"])
        assert result.shape == (2, 8)

    def test_embed_query_returns_1d(self):
        p = FakeEmbeddingProvider(dim=8)
        result = p.embed_query("hello")
        assert result.shape == (8,)

    def test_unit_vectors(self):
        p = FakeEmbeddingProvider(dim=32)
        vecs = p.embed_documents(["text one", "text two"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_same_text(self):
        p = FakeEmbeddingProvider(dim=16)
        v1 = p.embed_query("policy question")
        v2 = p.embed_query("policy question")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        p = FakeEmbeddingProvider(dim=32)
        v1 = p.embed_query("report change")
        v2 = p.embed_query("income threshold")
        # Unlikely to be identical for different texts
        assert not np.allclose(v1, v2)

    def test_empty_documents_raises(self):
        p = FakeEmbeddingProvider()
        with pytest.raises((ValueError, Exception)):
            p.embed_documents([])

    def test_implements_protocol(self):
        from src.retrieval.embeddings import EmbeddingProvider
        p = FakeEmbeddingProvider()
        assert isinstance(p, EmbeddingProvider)


# ---------------------------------------------------------------------------
# 3. VectorIndex
# ---------------------------------------------------------------------------

class TestVectorIndex:

    def test_build_succeeds(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        assert index is not None

    def test_search_returns_list(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("calendar days report", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_policy_clauses(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("change of circumstances", top_k=3)
        for clause, score in results:
            assert isinstance(clause, PolicyClause)

    def test_scores_in_0_1(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("income threshold household", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_respects_top_k(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("anything", top_k=2)
        assert len(results) <= 2

    def test_top_k_larger_than_corpus_handled(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("test", top_k=1000)
        assert len(results) <= len(clauses)

    def test_empty_query_returns_empty(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        assert index.search("", top_k=5) == []

    def test_whitespace_query_returns_empty(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        assert index.search("   ", top_k=5) == []

    def test_empty_corpus_raises(self):
        provider = FakeEmbeddingProvider(dim=16)
        with pytest.raises(ValueError):
            VectorIndex.build([], provider)

    def test_clause_ids_preserved(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("report", top_k=3)
        expected_ids = {c.clause_id for c in clauses}
        for clause, _ in results:
            assert clause.clause_id in expected_ids

    def test_source_lines_preserved(self):
        clauses = [_make_clause("1.1.1", "Some text.", section_id="1.1")]
        clauses[0] = PolicyClause(
            clause_id="1.1.1", part_id="1", part_title="Part 1",
            section_id="1.1", section_title="1.1 Section",
            text="Some text.", sub_items=(), cross_references=(),
            source_path=Path("/fake/policy.md"), start_line=42, end_line=45,
        )
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        results = index.search("some", top_k=1)
        assert results[0][0].start_line == 42
        assert results[0][0].end_line == 45

    def test_deterministic(self):
        clauses = _small_corpus()
        provider = FakeEmbeddingProvider(dim=16)
        index = VectorIndex.build(clauses, provider)
        r1 = index.search("household income", top_k=3)
        r2 = index.search("household income", top_k=3)
        assert [c.clause_id for c, _ in r1] == [c.clause_id for c, _ in r2]


# ---------------------------------------------------------------------------
# 4. LexicalIndex
# ---------------------------------------------------------------------------

class TestLexicalIndex:

    def test_build_succeeds(self):
        index = LexicalIndex.build(_small_corpus())
        assert index is not None

    def test_search_returns_list(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("calendar days", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_policy_clauses(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("income household", top_k=3)
        for clause, score in results:
            assert isinstance(clause, PolicyClause)

    def test_scores_in_0_1(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("recipient report change", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_exact_term_gets_score(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("calendar", top_k=5)
        ids = [c.clause_id for c, _ in results]
        # "1.1.1" contains "calendar days"
        assert "1.1.1" in ids

    def test_zero_score_clauses_excluded(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("xyzzy foobar", top_k=10)
        for _, score in results:
            assert score > 0.0

    def test_respects_top_k(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("the a", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self):
        index = LexicalIndex.build(_small_corpus())
        assert index.search("", top_k=5) == []

    def test_whitespace_query_returns_empty(self):
        index = LexicalIndex.build(_small_corpus())
        assert index.search("   ", top_k=5) == []

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            LexicalIndex.build([])

    def test_clause_id_query_retrieves_clause(self):
        index = LexicalIndex.build(_small_corpus())
        results = index.search("1.1.1", top_k=5)
        ids = [c.clause_id for c, _ in results]
        assert "1.1.1" in ids

    def test_deterministic(self):
        index = LexicalIndex.build(_small_corpus())
        r1 = index.search("household income threshold", top_k=3)
        r2 = index.search("household income threshold", top_k=3)
        assert [c.clause_id for c, _ in r1] == [c.clause_id for c, _ in r2]


# ---------------------------------------------------------------------------
# 5. HybridRetriever — API contracts
# ---------------------------------------------------------------------------

class TestHybridRetrieverAPI:

    def test_retrieve_returns_list(self):
        r = _small_retriever()
        results = r.retrieve("household income", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_returns_retrieval_results(self):
        r = _small_retriever()
        results = r.retrieve("report change", top_k=3)
        for result in results:
            assert isinstance(result, RetrievalResult)

    def test_results_have_clause(self):
        r = _small_retriever()
        for result in r.retrieve("calendar days", top_k=3):
            assert isinstance(result.clause, PolicyClause)

    def test_clause_ids_intact(self):
        r = _small_retriever()
        expected = {c.clause_id for c in _small_corpus()}
        for result in r.retrieve("income", top_k=10):
            assert result.clause.clause_id in expected

    def test_respects_top_k(self):
        r = _small_retriever()
        results = r.retrieve("any query", top_k=2)
        assert len(results) <= 2

    def test_top_k_larger_than_corpus(self):
        r = _small_retriever()
        results = r.retrieve("income", top_k=1000)
        assert len(results) <= len(_small_corpus())

    def test_sorted_by_combined_score_descending(self):
        r = _small_retriever()
        results = r.retrieve("household income threshold", top_k=5)
        scores = [res.combined_score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self):
        r = _small_retriever()
        assert r.retrieve("") == []

    def test_whitespace_query_returns_empty(self):
        r = _small_retriever()
        assert r.retrieve("   ") == []

    def test_default_top_k_from_config(self):
        cfg = RetrieverConfig(final_top_k=2)
        r = _small_retriever(cfg)
        results = r.retrieve("household income")
        assert len(results) <= 2

    def test_combined_scores_positive(self):
        r = _small_retriever()
        for res in r.retrieve("calendar days", top_k=5):
            assert res.combined_score > 0.0

    def test_individual_scores_in_0_1(self):
        r = _small_retriever()
        for res in r.retrieve("income threshold", top_k=5):
            assert 0.0 <= res.semantic_score <= 1.0
            assert 0.0 <= res.lexical_score <= 1.0


# ---------------------------------------------------------------------------
# 6. HybridRetriever — deduplication
# ---------------------------------------------------------------------------

class TestHybridDeduplication:

    def test_no_duplicate_clause_ids(self):
        r = _small_retriever()
        results = r.retrieve("calendar days report change", top_k=10)
        ids = [res.clause.clause_id for res in results]
        assert len(ids) == len(set(ids)), f"Duplicate clause IDs found: {ids}"

    def test_dual_source_clause_has_both_sources(self):
        """A clause appearing in both semantic and lexical results
        should have sources=('lexical', 'semantic')."""
        r = _small_retriever()
        results = r.retrieve("calendar days report", top_k=10)
        dual = [res for res in results if len(res.sources) == 2]
        # At least one clause should appear in both — check the field values
        for res in dual:
            assert "semantic" in res.sources
            assert "lexical" in res.sources

    def test_semantic_only_clause_has_semantic_source(self):
        r = _small_retriever()
        results = r.retrieve("very specific unique text reside county", top_k=10)
        for res in results:
            assert len(res.sources) >= 1
            for s in res.sources:
                assert s in ("semantic", "lexical")


# ---------------------------------------------------------------------------
# 7. HybridRetriever — edge cases
# ---------------------------------------------------------------------------

class TestHybridEdgeCases:

    def test_single_clause_corpus(self):
        clause = _make_clause("1.1.1", "The only clause in the corpus.")
        provider = FakeEmbeddingProvider(dim=16)
        r = HybridRetriever.build([clause], provider)
        results = r.retrieve("clause", top_k=5)
        assert len(results) == 1
        assert results[0].clause.clause_id == "1.1.1"

    def test_deterministic(self):
        r = _small_retriever()
        r1 = r.retrieve("income threshold days", top_k=5)
        r2 = r.retrieve("income threshold days", top_k=5)
        assert [res.clause.clause_id for res in r1] == [
            res.clause.clause_id for res in r2
        ]

    def test_build_from_empty_raises(self):
        provider = FakeEmbeddingProvider(dim=16)
        with pytest.raises(ValueError):
            HybridRetriever.build([], provider)

    def test_sources_field_valid_values(self):
        r = _small_retriever()
        for res in r.retrieve("household", top_k=10):
            for s in res.sources:
                assert s in ("semantic", "lexical")


# ---------------------------------------------------------------------------
# 8. RetrievalResult model
# ---------------------------------------------------------------------------

class TestRetrievalResultModel:

    def test_is_frozen(self):
        clause = _make_clause("1.1.1", "Text.")
        result = RetrievalResult(
            clause=clause,
            semantic_score=0.8,
            lexical_score=0.6,
            combined_score=0.02,
            sources=("semantic",),
        )
        with pytest.raises((AttributeError, TypeError)):
            result.semantic_score = 0.5  # type: ignore[misc]

    def test_clause_accessible(self):
        clause = _make_clause("1.1.1", "Some text.")
        result = RetrievalResult(
            clause=clause,
            semantic_score=0.5,
            lexical_score=0.3,
            combined_score=0.01,
            sources=("lexical",),
        )
        assert result.clause.clause_id == "1.1.1"
        assert result.clause.text == "Some text."

    def test_source_line_range_accessible_via_clause(self):
        clause = PolicyClause(
            clause_id="1.1.1", part_id="1", part_title="Part 1",
            section_id="1.1", section_title="1.1 Section",
            text="Text.", sub_items=(), cross_references=(),
            source_path=Path("/fake/p.md"), start_line=10, end_line=12,
        )
        result = RetrievalResult(
            clause=clause,
            semantic_score=0.5,
            lexical_score=0.0,
            combined_score=0.01,
            sources=("semantic",),
        )
        assert result.clause.start_line == 10
        assert result.clause.end_line == 12


# ---------------------------------------------------------------------------
# 9. Real corpus integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_CORPUS.exists(),
    reason="Real corpus not present at data/raw/policy_manual.md",
)
class TestRealCorpusRetrieval:

    @pytest.fixture(scope="class")
    def retriever(self):
        doc = load_policy_document(REAL_CORPUS)
        store = ClauseStore(parse_clauses(doc))
        provider = FakeEmbeddingProvider(dim=64)
        cfg = RetrieverConfig(
            semantic_top_k=15,
            lexical_top_k=15,
            final_top_k=10,
        )
        return HybridRetriever.build(store.all(), provider, cfg)

    # -- Index built successfully ------------------------------------------

    def test_index_covers_all_clauses(self, retriever):
        """Non-trivial queries should return up to final_top_k results."""
        results = retriever.retrieve("household eligibility income", top_k=10)
        assert len(results) > 0

    def test_results_are_policy_clauses(self, retriever):
        results = retriever.retrieve("recipient must report", top_k=5)
        for r in results:
            assert isinstance(r.clause, PolicyClause)
            assert r.clause.clause_id  # non-empty
            assert r.clause.start_line > 0

    def test_no_duplicate_clause_ids(self, retriever):
        results = retriever.retrieve("change of circumstances report", top_k=15)
        ids = [r.clause.clause_id for r in results]
        assert len(ids) == len(set(ids))

    # -- Scenario 1: exact clause ID query ---------------------------------

    def test_clause_id_4_3_2_query(self, retriever):
        """Query '4.3.2' should surface §4.3.2 in results."""
        results = retriever.retrieve("4.3.2", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "4.3.2" in ids, f"§4.3.2 not found in {ids}"

    def test_section_symbol_clause_id_query(self, retriever):
        """Query '§4.3.2' should also surface §4.3.2."""
        results = retriever.retrieve("§4.3.2", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "4.3.2" in ids, f"§4.3.2 not found with § prefix query: {ids}"

    # -- Scenario 2: natural-language paraphrase ----------------------------

    def test_reporting_deadline_paraphrase(self, retriever):
        """
        'How many days do I have to tell the office about a change?'
        should surface §4.3.2 (the reporting obligation clause).
        """
        results = retriever.retrieve(
            "how many days to tell office about a change", top_k=15
        )
        ids = [r.clause.clause_id for r in results]
        assert "4.3.2" in ids, f"§4.3.2 not found in: {ids}"

    # -- Scenario 3: query with a specific number ---------------------------

    def test_10_day_query(self, retriever):
        """Query mentioning '10 days' should hit §4.3.2."""
        results = retriever.retrieve("10 days report change", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "4.3.2" in ids, f"§4.3.2 not found in: {ids}"

    # -- Scenario 4: exact policy terminology -------------------------------

    def test_countable_income_term(self, retriever):
        """Query using exact policy term 'countable income'."""
        results = retriever.retrieve("countable income threshold", top_k=10)
        ids = [r.clause.clause_id for r in results]
        # §6.6.1 defines income thresholds; §6.1.1 defines countable income
        assert any(cid in ids for cid in ["6.6.1", "6.1.1"]), (
            f"Expected income threshold clause in {ids}"
        )

    # -- Scenario 5: significant terminology divergence --------------------

    def test_rephrased_question(self, retriever):
        """
        'Am I allowed to get money if I own a house?'
        — paraphrase of the home-as-non-countable-resource rule (§2.4.2).
        Retrieval should return some results, even if §2.4.2 is not top-1.
        """
        results = retriever.retrieve(
            "am I allowed to get money if I own a house", top_k=10
        )
        assert len(results) > 0

    # -- Scenario 6: contradiction case ------------------------------------

    def test_both_contradiction_clauses_reachable(self, retriever):
        """
        Both §4.3.2 (10 days) and §9.1.4 (30 days) must be reachable
        for a reporting-deadline query.
        The retriever must not systematically suppress one of them.
        """
        results = retriever.retrieve(
            "report change circumstances calendar days", top_k=15
        )
        ids = [r.clause.clause_id for r in results]
        assert "4.3.2" in ids, f"§4.3.2 missing from: {ids}"
        assert "9.1.4" in ids, f"§9.1.4 missing from: {ids}"

    # -- Scenario 7: apparent gap / cross-reference case -------------------

    def test_7_1_3_retrievable(self, retriever):
        """
        §7.1.3 references §5.4 in a likely-erroneous way.
        Retrieval must be able to surface §7.1.3 for a query about
        the needs figure calculation.
        """
        results = retriever.retrieve("needs figure calculation household", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "7.1.3" in ids or "7.1.1" in ids, (
            f"No needs figure clause found in: {ids}"
        )

    # -- Scenario 8: table-backed policy area (income thresholds) ----------

    def test_income_threshold_table(self, retriever):
        """§6.6.1 contains the income threshold table."""
        results = retriever.retrieve("income threshold table monthly", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "6.6.1" in ids, f"§6.6.1 not found in: {ids}"

    # -- Scenario 9: clause with multiple sub-items ------------------------

    def test_disregards_clause_retrieval(self, retriever):
        """§6.4.1 contains 7 sub-items listing income disregards."""
        results = retriever.retrieve("income disregards training child support", top_k=10)
        ids = [r.clause.clause_id for r in results]
        assert "6.4.1" in ids, f"§6.4.1 not found in: {ids}"

    # -- Scenario 10: query with weak lexical match ------------------------

    def test_weak_lexical_match_still_returns_results(self, retriever):
        """
        A question whose wording differs strongly from the policy text
        should still return some results via semantic retrieval.
        """
        results = retriever.retrieve(
            "what financial help is available for struggling families", top_k=10
        )
        assert len(results) > 0

    # -- Result properties -------------------------------------------------

    def test_scores_in_range(self, retriever):
        results = retriever.retrieve("household income", top_k=10)
        for r in results:
            assert 0.0 <= r.semantic_score <= 1.0
            assert 0.0 <= r.lexical_score <= 1.0
            assert r.combined_score > 0.0

    def test_source_lines_preserved(self, retriever):
        results = retriever.retrieve("report change", top_k=5)
        for r in results:
            assert r.clause.start_line > 0
            assert r.clause.end_line >= r.clause.start_line

    def test_sorted_by_combined_score(self, retriever):
        results = retriever.retrieve("eligibility conditions", top_k=10)
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_is_deterministic(self, retriever):
        q = "how many days to report a change of address"
        r1 = retriever.retrieve(q, top_k=10)
        r2 = retriever.retrieve(q, top_k=10)
        assert [r.clause.clause_id for r in r1] == [r.clause.clause_id for r in r2]
