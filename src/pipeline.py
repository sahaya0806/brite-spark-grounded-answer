"""
End-to-end policy question answering pipeline with temporal applicability.

Architecture
------------
Question + TemporalContext (optional)
   ↓
HybridRetriever (Semantic + BM25)
   ↓
Candidate Clauses (RetrievalResult)
   ↓
TemporalFilter (Resolve applicable policy versions / flag missing date)
   ↓
EvidenceEvaluator (SUPPORTED / INSUFFICIENT / CONFLICTING)
   ↓
GroundedAnswerGenerator (Grounded plain-language answer + exact citations)
"""

from __future__ import annotations

from datetime import date as dt_date
from pathlib import Path

from src.evidence.evaluator import EvidenceConfig, EvidenceEvaluator
from src.evidence.models import DecisionStatus, EvidenceDecision
from src.generation.generator import GroundedAnswerGenerator
from src.generation.models import GroundedAnswer
from src.generation.providers import ChatProvider, OpenAIChatProvider
from src.ingestion.amendment import parse_amendment
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import parse_clauses
from src.ingestion.store import ClauseStore
from src.retrieval.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from src.retrieval.hybrid import HybridRetriever, RetrieverConfig
from src.temporal import (
    TemporalApplicabilityResolver,
    TemporalContext,
    TemporalFilter,
)


class PolicyQAPipeline:
    """
    Unified end-to-end pipeline for grounded policy question answering.

    Parameters
    ----------
    retriever:
        Configured HybridRetriever instance.
    evaluator:
        Configured EvidenceEvaluator instance.
    generator:
        Configured GroundedAnswerGenerator instance.
    temporal_filter:
        Optional TemporalFilter instance for date-aware policy resolution.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        evaluator: EvidenceEvaluator | None = None,
        generator: GroundedAnswerGenerator | None = None,
        temporal_filter: TemporalFilter | None = None,
    ) -> None:
        self.retriever = retriever
        self.evaluator = evaluator or EvidenceEvaluator()
        self.generator = generator or GroundedAnswerGenerator()
        self.temporal_filter = temporal_filter

    def ask(
        self,
        question: str,
        context: TemporalContext | None = None,
        date: dt_date | str | None = None,
        top_k: int = 10,
    ) -> GroundedAnswer:
        """
        Execute the complete RAG pipeline on a question.

        1. Retrieve candidate clauses via hybrid search.
        2. Apply temporal filtering to select applicable policy versions.
        3. Evaluate evidence sufficiency, gaps, and contradictions.
        4. Synthesize a grounded answer or explicit refusal/conflict explanation.

        Parameters
        ----------
        question:
            The user's policy question.
        context:
            Explicit TemporalContext with determination date and/or change date.
        date:
            Convenience date parameter (YYYY-MM-DD or datetime.date).
        top_k:
            Number of retrieval candidates to consider. Default 10.

        Returns
        -------
        GroundedAnswer
            Final structured answer with status, text, and citations.
        """
        # Resolve convenience date into TemporalContext if provided
        temporal_ctx = context
        if date is not None:
            parsed_d = dt_date.fromisoformat(date) if isinstance(date, str) else date
            if temporal_ctx is None:
                temporal_ctx = TemporalContext(
                    determination_date=parsed_d,
                    change_of_circumstances_date=parsed_d,
                    claim_date=parsed_d,
                )
            else:
                # Merge date into empty fields
                temporal_ctx = TemporalContext(
                    determination_date=temporal_ctx.determination_date or parsed_d,
                    change_of_circumstances_date=temporal_ctx.change_of_circumstances_date or parsed_d,
                    claim_date=temporal_ctx.claim_date or parsed_d,
                )

        # 1. Retrieve candidates
        raw_results = self.retriever.retrieve(question, top_k=top_k)

        # 2. Temporal filtering
        unresolved_top: TemporalResolution | None = None

        if self.temporal_filter is not None:
            filter_res = self.temporal_filter.filter_with_status(raw_results, temporal_ctx)
            eval_results = list(filter_res.results)

            if filter_res.unresolved_clauses:
                unresolved_map = {u.clause_id: u for u in filter_res.unresolved_clauses}
                # Check if any top retrieval candidate requires temporal context
                for r in raw_results[:3]:
                    if r.clause.clause_id in unresolved_map:
                        unresolved_top = unresolved_map[r.clause.clause_id]
                        break
        else:
            eval_results = raw_results

        # If the primary retrieved clause requires temporal context that was not provided, refuse
        if unresolved_top is not None:
            missing_cid = unresolved_top.clause_id
            change = unresolved_top.amendment_change
            amend_id = change.amendment_id if change else "2026-01"
            eff_date = change.effective_from.isoformat() if change else "2026-03-01"
            base_clause = next((r.clause for r in raw_results if r.clause.clause_id == missing_cid), None)
            primary = (base_clause,) if base_clause else ()

            decision = EvidenceDecision(
                status=DecisionStatus.INSUFFICIENT,
                question=question,
                evidence=(),
                rationale=(
                    f"Clause §{missing_cid} was amended by Amendment No. {amend_id} effective {eff_date}. "
                    f"A date parameter (--date YYYY-MM-DD) is required to determine whether the pre-amendment "
                    f"or post-amendment policy version applies."
                ),
                support_score=0.0,
                primary_clauses=primary,
                conflict_details=(),
                missing_information=(
                    f"A relevant date (--date YYYY-MM-DD) is required to determine whether the pre-amendment "
                    f"or Amendment No. {amend_id} policy version applies for §{missing_cid}."
                ),
                recommended_action="refuse",
            )
            return self.generator.generate_answer(decision)

        # 3. Evidence evaluation
        decision = self.evaluator.evaluate(question, eval_results)

        # 4. Grounded answer generation
        return self.generator.generate_answer(decision)

    @classmethod
    def build_from_corpus(
        cls,
        corpus_path: Path | str,
        amendment_path: Path | str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chat_provider: ChatProvider | None = None,
        retriever_config: RetrieverConfig | None = None,
        evidence_config: EvidenceConfig | None = None,
    ) -> PolicyQAPipeline:
        """
        Factory to construct a complete pipeline from a raw policy Markdown file
        and optional amendment Markdown file.

        Parameters
        ----------
        corpus_path:
            Path to the policy manual Markdown file.
        amendment_path:
            Optional path to amendment Markdown file. Defaults to None (base corpus only).
        embedding_provider:
            EmbeddingProvider (defaults to OpenAIEmbeddingProvider).
        chat_provider:
            ChatProvider (defaults to OpenAIChatProvider).
        retriever_config:
            Optional RetrieverConfig.
        evidence_config:
            Optional EvidenceConfig.
        """
        doc = load_policy_document(corpus_path)
        clauses = parse_clauses(doc)

        temporal_filter = None
        index_clauses = list(clauses)

        # Check if amendment path is provided and exists
        if amendment_path is not None:
            p = Path(amendment_path)
            if p.exists():
                amend_doc = load_policy_document(p)
                amendment = parse_amendment(amend_doc)
                resolver = TemporalApplicabilityResolver(clauses, amendment)
                temporal_filter = TemporalFilter(resolver)

                # Include any new clauses (e.g. §10.5.3A) in the retrieval index
                orig_ids = {c.clause_id for c in clauses}
                for amended_c in amendment.create_amended_clauses(clauses):
                    if amended_c.clause_id not in orig_ids:
                        index_clauses.append(amended_c)

        store = ClauseStore(index_clauses)
        emb_provider = embedding_provider or OpenAIEmbeddingProvider()
        retriever = HybridRetriever.build(
            store.all(),
            emb_provider,
            config=retriever_config,
        )

        evaluator = EvidenceEvaluator(config=evidence_config)
        generator = GroundedAnswerGenerator(provider=chat_provider)

        return cls(
            retriever=retriever,
            evaluator=evaluator,
            generator=generator,
            temporal_filter=temporal_filter,
        )

