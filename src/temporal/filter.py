"""
Temporal filtering of retrieved policy evidence.

Responsibilities:
- Filter and transform RetrievalResults according to TemporalContext.
- Select applicable PolicyClause version (original vs amended) for each candidate.
- Filter out non-applicable or not-yet-in-force provisions (e.g. §10.5.3A pre-amendment).
- Preserve retrieval scores, rank ordering, and source methods.
- Identify when date-sensitive clauses require missing temporal context.

Design principles:
- Deterministic and explainable.
- Never mutates the original PolicyClause objects.
- Temporal filtering occurs BEFORE evidence evaluation and contradiction detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.ingestion.parser import PolicyClause
from src.retrieval.models import RetrievalResult
from src.temporal.models import (
    ResolutionStatus,
    TemporalContext,
    TemporalResolution,
)
from src.temporal.resolver import TemporalApplicabilityResolver


@dataclass(frozen=True)
class TemporalFilterResult:
    """
    Structured outcome of temporal filtering across a list of RetrievalResults.

    Attributes
    ----------
    results:
        Filtered and updated RetrievalResult objects whose clauses match the temporal context.
    resolutions:
        Tuple of per-clause TemporalResolution objects in candidate order.
    unresolved_clauses:
        Resolutions where status == TEMPORAL_CONTEXT_REQUIRED.
    """
    results: tuple[RetrievalResult, ...]
    resolutions: tuple[TemporalResolution, ...]
    unresolved_clauses: tuple[TemporalResolution, ...]

    @property
    def requires_context(self) -> bool:
        """True if any highly-relevant candidate clause requires missing temporal context."""
        return bool(self.unresolved_clauses)


class TemporalFilter:
    """
    Filters and adapts retrieval results to match the applicable temporal context.

    Parameters
    ----------
    resolver:
        Configured TemporalApplicabilityResolver.
    """

    def __init__(self, resolver: TemporalApplicabilityResolver) -> None:
        self._resolver = resolver

    @property
    def resolver(self) -> TemporalApplicabilityResolver:
        return self._resolver

    def filter_with_status(
        self,
        results: Sequence[RetrievalResult],
        context: TemporalContext | None = None,
    ) -> TemporalFilterResult:
        """
        Filter and update RetrievalResults against the temporal context.

        Parameters
        ----------
        results:
            List of RetrievalResult objects from HybridRetriever.
        context:
            TemporalContext with determination date and/or change of circumstances date.

        Returns
        -------
        TemporalFilterResult
            Structured result containing active RetrievalResults and any unresolved requirements.
        """
        filtered_results: list[RetrievalResult] = []
        resolutions: list[TemporalResolution] = []
        unresolved: list[TemporalResolution] = []

        for item in results:
            cid = item.clause.clause_id
            res = self._resolver.resolve(cid, context)
            resolutions.append(res)

            if res.is_resolved and res.applicable_clause is not None:
                # Replace clause with resolved version, preserving retrieval scores and sources
                updated_item = RetrievalResult(
                    clause=res.applicable_clause,
                    semantic_score=item.semantic_score,
                    lexical_score=item.lexical_score,
                    combined_score=item.combined_score,
                    sources=item.sources,
                )
                filtered_results.append(updated_item)
            elif res.status == ResolutionStatus.TEMPORAL_CONTEXT_REQUIRED:
                unresolved.append(res)
            elif res.status == ResolutionStatus.NOT_YET_IN_FORCE:
                # Clause was not yet in force at query date — omit from active evidence
                continue
            elif res.status == ResolutionStatus.CLAUSE_NOT_FOUND:
                continue

        return TemporalFilterResult(
            results=tuple(filtered_results),
            resolutions=tuple(resolutions),
            unresolved_clauses=tuple(unresolved),
        )

    def filter(
        self,
        results: Sequence[RetrievalResult],
        context: TemporalContext | None = None,
    ) -> list[RetrievalResult]:
        """
        Convenience method returning the list of active RetrievalResults.
        """
        return list(self.filter_with_status(results, context).results)
