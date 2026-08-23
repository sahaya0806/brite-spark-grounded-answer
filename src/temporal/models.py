"""
Temporal context and policy version resolution models.

Responsibilities:
- Represent date context supplied for a query (determination date vs change of circumstances date).
- Represent resolution status (RESOLVED_ORIGINAL, RESOLVED_AMENDMENT, UNAMENDED, TEMPORAL_CONTEXT_REQUIRED, NOT_YET_IN_FORCE, CLAUSE_NOT_FOUND).
- Represent detailed resolution result with citable clause and explanation.

Design principles:
- Immutable dataclasses.
- Strict typing with standard library datetime.date.
- Never silently guess or substitute date types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Sequence

from src.ingestion.amendment import AmendmentChange, TriggerType
from src.ingestion.parser import PolicyClause


# ---------------------------------------------------------------------------
# Temporal Context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalContext:
    """
    Temporal parameters associated with a policy question or benefit claim.

    Attributes
    ----------
    determination_date:
        The date on which the caseworker or Department makes the determination.
        Governs amendments under §5.1 of Amendment No. 2026-01 (earnings disregard,
        income thresholds, sanctions).
    change_of_circumstances_date:
        The date on which the relevant change in circumstances occurred.
        Governs amendments under §5.2 of Amendment No. 2026-01 (reporting period
        for §4.3.2 and §9.1.4).
    claim_date:
        Optional general claim or event reference date (e.g. claim period start).
    """
    determination_date: date | None = None
    change_of_circumstances_date: date | None = None
    claim_date: date | None = None

    @property
    def is_empty(self) -> bool:
        """True if no temporal dates are supplied."""
        return (
            self.determination_date is None
            and self.change_of_circumstances_date is None
            and self.claim_date is None
        )

    def has_date_for(self, trigger_type: TriggerType) -> bool:
        """Check if context contains a date for the specified trigger type."""
        if trigger_type == TriggerType.DETERMINATION_DATE:
            return self.determination_date is not None
        elif trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE:
            return self.change_of_circumstances_date is not None
        elif trigger_type == TriggerType.CLAIM_PERIOD:
            return self.claim_date is not None or self.determination_date is not None
        return False

    def get_date_for(self, trigger_type: TriggerType) -> date | None:
        """Retrieve the exact date corresponding to the trigger type without fallback."""
        if trigger_type == TriggerType.DETERMINATION_DATE:
            return self.determination_date
        elif trigger_type == TriggerType.CHANGE_OF_CIRCUMSTANCES_DATE:
            return self.change_of_circumstances_date
        elif trigger_type == TriggerType.CLAIM_PERIOD:
            return self.claim_date or self.determination_date
        return None


# ---------------------------------------------------------------------------
# Resolution Status & Result
# ---------------------------------------------------------------------------

class ResolutionStatus(str, Enum):
    """
    Outcome of resolving a clause against a temporal context.

    RESOLVED_ORIGINAL
        The original manual's policy clause version applies for this date.

    RESOLVED_AMENDMENT
        The amended policy clause version applies for this date.

    UNAMENDED
        The clause was never amended and is active across all dates.

    TEMPORAL_CONTEXT_REQUIRED
        The clause was modified by an amendment, but the required date type
        (e.g. determination date vs change of circumstances date) is missing
        from TemporalContext. The system must refuse rather than guess.

    NOT_YET_IN_FORCE
        The provision is a new insertion (e.g. §10.5.3A) that was not yet in force
        at the specified historical date.

    CLAUSE_NOT_FOUND
        The requested clause ID does not exist in either base manual or amendments.
    """
    RESOLVED_ORIGINAL = "resolved_original"
    RESOLVED_AMENDMENT = "resolved_amendment"
    UNAMENDED = "unamended"
    TEMPORAL_CONTEXT_REQUIRED = "temporal_context_required"
    NOT_YET_IN_FORCE = "not_yet_in_force"
    CLAUSE_NOT_FOUND = "clause_not_found"


@dataclass(frozen=True)
class TemporalResolution:
    """
    Detailed outcome of resolving a clause version against temporal context.

    Attributes
    ----------
    clause_id:
        Identifier of the evaluated clause (e.g. "4.3.2", "6.4.1", "10.5.3A").
    status:
        The resolution outcome (RESOLVED_ORIGINAL, RESOLVED_AMENDMENT, etc.).
    applicable_clause:
        The authoritative PolicyClause instance if resolved, or None if context
        is missing / clause not in force.
    amendment_change:
        Associated AmendmentChange if this clause is subject to an amendment.
    source_document:
        Filename of the governing source document ("policy_manual.md" or amendment file).
    trigger_type:
        The temporal trigger governing applicability, if amended.
    reason:
        Human-readable explanation of why this version was selected or why
        additional context is required.
    """
    clause_id: str
    status: ResolutionStatus
    applicable_clause: PolicyClause | None
    amendment_change: AmendmentChange | None
    source_document: str
    trigger_type: TriggerType | None
    reason: str

    @property
    def is_resolved(self) -> bool:
        """True if an authoritative PolicyClause was successfully resolved."""
        return self.status in (
            ResolutionStatus.RESOLVED_ORIGINAL,
            ResolutionStatus.RESOLVED_AMENDMENT,
            ResolutionStatus.UNAMENDED,
        ) and self.applicable_clause is not None
