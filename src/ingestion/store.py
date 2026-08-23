"""
In-memory structured clause store.

Provides a simple, typed interface for retrieving parsed policy clauses.
No database, no vector index — just a list wrapped in a small API surface
that future retrieval code can depend on without coupling to the raw list.
"""

from __future__ import annotations

from src.ingestion.parser import PolicyClause


class ClauseNotFoundError(KeyError):
    """Raised when a clause ID is not present in the store."""


class ClauseStore:
    """
    An in-memory store of parsed policy clauses.

    Build it from the output of ``parse_clauses``:

        store = ClauseStore(parse_clauses(document))

    Then query:

        clause = store.get_by_id("4.3.2")
        all_clauses = store.all()
        n = store.count()
    """

    def __init__(self, clauses: list[PolicyClause]) -> None:
        self._clauses: list[PolicyClause] = list(clauses)
        self._index: dict[str, PolicyClause] = {
            c.clause_id: c for c in self._clauses
        }

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_by_id(self, clause_id: str) -> PolicyClause:
        """
        Return the clause with *clause_id*.

        Parameters
        ----------
        clause_id:
            Numeric ID without the § prefix, e.g. ``"4.3.2"``.

        Raises
        ------
        ClauseNotFoundError
            If no clause with that ID exists in the store.
        """
        try:
            return self._index[clause_id]
        except KeyError:
            raise ClauseNotFoundError(
                f"Clause not found in store: §{clause_id!r}"
            ) from None

    def all(self) -> list[PolicyClause]:
        """Return all clauses in document order."""
        return list(self._clauses)

    def count(self) -> int:
        """Return the number of clauses in the store."""
        return len(self._clauses)

    def __repr__(self) -> str:
        return f"ClauseStore({self.count()} clauses)"
