"""
Deterministic temporal applicability resolver for versioned policy clauses.

Responsibilities:
- Determine which policy clause version applies given a clause ID and TemporalContext.
- Enforce exact temporal triggers defined by amendments (determination date vs change of circumstances date).
- Refuse with TEMPORAL_CONTEXT_REQUIRED when a date-sensitive clause lacks required temporal context.
- Distinguish pre-amendment original versions, post-amendment versions, and unamended provisions.
- Provide full auditability and explanation of version resolution.

Design principles:
- 100% deterministic (no LLMs, no guessing, no fallbacks).
- Immutability: original policy documents and clause objects are never mutated.
"""

from __future__ import annotations

from typing import Sequence

from src.ingestion.amendment import (
    AmendmentChange,
    AmendmentDocument,
    ChangeType,
    TriggerType,
)
from src.ingestion.parser import PolicyClause
from src.temporal.models import (
    ResolutionStatus,
    TemporalContext,
    TemporalResolution,
)


class TemporalApplicabilityResolver:
    """
    Evaluates temporal applicability of policy clauses against a TemporalContext.

    Parameters
    ----------
    original_clauses:
        All base policy clauses extracted from the original policy manual.
    amendment:
        Parsed AmendmentDocument containing structured amendment changes, if loaded.
    """

    def __init__(
        self,
        original_clauses: Sequence[PolicyClause],
        amendment: AmendmentDocument | None = None,
    ) -> None:
        self._original_clauses = list(original_clauses)
        self._original_by_id: dict[str, PolicyClause] = {
            c.clause_id: c for c in original_clauses
        }
        self._amendment = amendment
        self._changes_by_id: dict[str, AmendmentChange] = {}
        self._amended_clauses_by_id: dict[str, PolicyClause] = {}

        if amendment is not None:
            for change in amendment.changes:
                self._changes_by_id[change.target_clause_id] = change

            amended_clauses = amendment.create_amended_clauses(self._original_clauses)
            for clause in amended_clauses:
                self._amended_clauses_by_id[clause.clause_id] = clause

    @property
    def amendment(self) -> AmendmentDocument | None:
        return self._amendment

    def is_date_sensitive(self, clause_id: str) -> bool:
        """Return True if the clause is affected or introduced by an amendment."""
        return clause_id in self._changes_by_id

    def get_required_trigger(self, clause_id: str) -> TriggerType | None:
        """Return the temporal trigger type required to resolve this clause, or None if unamended."""
        change = self._changes_by_id.get(clause_id)
        return change.trigger_type if change else None

    def resolve(
        self,
        clause_id: str,
        context: TemporalContext | None = None,
    ) -> TemporalResolution:
        """
        Determine the applicable version of a policy clause under the given temporal context.

        Parameters
        ----------
        clause_id:
            Numeric or alphanumeric clause identifier (e.g. "4.3.2", "6.4.1", "10.5.3A").
        context:
            Temporal context containing determination date and/or change of circumstances date.

        Returns
        -------
        TemporalResolution
            Detailed resolution with status, applicable clause, and explanation.
        """
        # Case 1: Unknown clause ID
        if clause_id not in self._original_by_id and clause_id not in self._amended_clauses_by_id:
            return TemporalResolution(
                clause_id=clause_id,
                status=ResolutionStatus.CLAUSE_NOT_FOUND,
                applicable_clause=None,
                amendment_change=None,
                source_document="",
                trigger_type=None,
                reason=f"Clause §{clause_id} does not exist in policy corpus.",
            )

        # Case 2: Unamended clause
        if clause_id not in self._changes_by_id:
            orig = self._original_by_id[clause_id]
            return TemporalResolution(
                clause_id=clause_id,
                status=ResolutionStatus.UNAMENDED,
                applicable_clause=orig,
                amendment_change=None,
                source_document="policy_manual.md",
                trigger_type=None,
                reason=f"Clause §{clause_id} has not been amended and remains active across all dates.",
            )

        # Case 3: Amended clause
        change = self._changes_by_id[clause_id]
        trigger = change.trigger_type
        effective_from = change.effective_from

        # Check whether the required temporal context was provided
        if context is None or not context.has_date_for(trigger):
            return TemporalResolution(
                clause_id=clause_id,
                status=ResolutionStatus.TEMPORAL_CONTEXT_REQUIRED,
                applicable_clause=None,
                amendment_change=change,
                source_document="policy_manual.md",
                trigger_type=trigger,
                reason=(
                    f"Clause §{clause_id} was amended by Amendment No. {change.amendment_id} "
                    f"effective {effective_from.isoformat()} based on {trigger.value}. "
                    f"A valid {trigger.value} must be provided to determine the applicable version."
                ),
            )

        query_date = context.get_date_for(trigger)
        assert query_date is not None

        # Compare against amendment effective date
        if query_date < effective_from:
            # Pre-amendment period
            if change.change_type == ChangeType.INSERTION:
                # New clause did not exist yet
                return TemporalResolution(
                    clause_id=clause_id,
                    status=ResolutionStatus.NOT_YET_IN_FORCE,
                    applicable_clause=None,
                    amendment_change=change,
                    source_document=change.source_document,
                    trigger_type=trigger,
                    reason=(
                        f"Clause §{clause_id} was inserted by Amendment No. {change.amendment_id} "
                        f"effective {effective_from.isoformat()} and was not in force on "
                        f"{query_date.isoformat()}."
                    ),
                )
            else:
                # Original base manual clause applies
                orig = self._original_by_id[clause_id]
                return TemporalResolution(
                    clause_id=clause_id,
                    status=ResolutionStatus.RESOLVED_ORIGINAL,
                    applicable_clause=orig,
                    amendment_change=change,
                    source_document="policy_manual.md",
                    trigger_type=trigger,
                    reason=(
                        f"On {query_date.isoformat()} ({trigger.value}), the original policy provision "
                        f"applies prior to Amendment No. {change.amendment_id} effective date {effective_from.isoformat()}."
                    ),
                )
        else:
            # Post-amendment period
            amended_clause = self._amended_clauses_by_id[clause_id]
            return TemporalResolution(
                clause_id=clause_id,
                status=ResolutionStatus.RESOLVED_AMENDMENT,
                applicable_clause=amended_clause,
                amendment_change=change,
                source_document=change.source_document,
                trigger_type=trigger,
                reason=(
                    f"On {query_date.isoformat()} ({trigger.value}), Amendment No. {change.amendment_id} "
                    f"applies (in force from {effective_from.isoformat()})."
                ),
            )

    def resolve_all(
        self,
        context: TemporalContext | None = None,
    ) -> list[TemporalResolution]:
        """
        Resolve all policy provisions against the given temporal context.

        Returns
        -------
        list[TemporalResolution]
            Resolution for every base clause and new amendment clause in order.
        """
        # Collect all unique clause IDs in original order, then append new clauses
        clause_ids: list[str] = [c.clause_id for c in self._original_clauses]
        for cid in self._amended_clauses_by_id:
            if cid not in self._original_by_id:
                clause_ids.append(cid)

        return [self.resolve(cid, context) for cid in clause_ids]

    def get_active_clauses(
        self,
        context: TemporalContext | None = None,
    ) -> list[PolicyClause]:
        """
        Return the list of active PolicyClause instances that are resolved for the context.

        Any clause requiring context that is missing will not be included.
        """
        resolutions = self.resolve_all(context)
        return [r.applicable_clause for r in resolutions if r.is_resolved and r.applicable_clause is not None]
