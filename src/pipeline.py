"""
End-to-end policy question answering pipeline.

Architecture
------------
Question
   ↓
HybridRetriever (Semantic + BM25)
   ↓
EvidenceEvaluator (SUPPORTED / INSUFFICIENT / CONFLICTING)
   ↓
GroundedAnswerGenerator (Grounded plain-language answer + exact citations)
"""

from __future__ import annotations

from pathlib import Path
from src.evidence.evaluator import EvidenceConfig, EvidenceEvaluator
from src.generation.generator import GroundedAnswerGenerator
from src.generation.models import GroundedAnswer
from src.generation.providers import ChatProvider, OpenAIChatProvider
from src.ingestion.loader import load_policy_document
from src.ingestion.parser import parse_clauses
from src.ingestion.store import ClauseStore
from src.retrieval.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from src.retrieval.hybrid import HybridRetriever, RetrieverConfig


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
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        evaluator: EvidenceEvaluator | None = None,
        generator: GroundedAnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.evaluator = evaluator or EvidenceEvaluator()
        self.generator = generator or GroundedAnswerGenerator()

    def ask(self, question: str, top_k: int = 10) -> GroundedAnswer:
        """
        Execute the complete RAG pipeline on a question.

        1. Retrieve candidate clauses via hybrid search.
        2. Evaluate evidence sufficiency, gaps, and contradictions.
        3. Synthesize a grounded answer or explicit refusal/conflict explanation.

        Parameters
        ----------
        question:
            The user's policy question.
        top_k:
            Number of retrieval candidates to consider. Default 10.

        Returns
        -------
        GroundedAnswer
            Final structured answer with status, text, and citations.
        """
        results = self.retriever.retrieve(question, top_k=top_k)
        decision = self.evaluator.evaluate(question, results)
        return self.generator.generate_answer(decision)

    @classmethod
    def build_from_corpus(
        cls,
        corpus_path: Path | str,
        embedding_provider: EmbeddingProvider | None = None,
        chat_provider: ChatProvider | None = None,
        retriever_config: RetrieverConfig | None = None,
        evidence_config: EvidenceConfig | None = None,
    ) -> PolicyQAPipeline:
        """
        Factory to construct a complete pipeline from a raw policy Markdown file.

        Parameters
        ----------
        corpus_path:
            Path to the policy manual Markdown file.
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
        store = ClauseStore(clauses)

        emb_provider = embedding_provider or OpenAIEmbeddingProvider()
        retriever = HybridRetriever.build(
            store.all(),
            emb_provider,
            config=retriever_config,
        )

        evaluator = EvidenceEvaluator(config=evidence_config)
        generator = GroundedAnswerGenerator(provider=chat_provider)

        return cls(retriever=retriever, evaluator=evaluator, generator=generator)
