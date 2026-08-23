"""
Clause-level parser for the Calder County Household Support Program policy manual.

Design basis
------------
This parser was designed from direct inspection of ``data/raw/policy_manual.md``
during Milestone 2.  The key structural facts that drive every design decision are:

1.  Authoritative clauses are identified by **bold paragraph openers** of the
    form ``**N.N.N**`` at the start of a line (regex ``^\\*\\*(\\d+\\.\\d+\\.\\d+)\\*\\*``).
    There are exactly 137 such openers in the real corpus.

2.  Document hierarchy uses two heading levels only:
    - H1 (``# …``) for Part headings after the three-line title block.
    - H2 (``## …``) for numbered sections (e.g. ``## 4.3 Recipient obligations``).

3.  Lettered sub-items ``(a) … (b) … (c) …`` belong to their parent clause.
    They are *not* independent clauses and must not receive their own IDs.

4.  Tables are embedded inside the body of the clause that introduces them
    (e.g. ``**6.6.1** … The thresholds are —`` followed immediately by table rows).
    They are accumulated as part of that clause's raw text.

5.  Cross-references use the form ``§N.N.N`` or ``§N.N`` throughout the text.

Nothing in this module modifies or interprets policy content.
Contradictions and apparent gaps in the source are preserved exactly as written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence

from src.ingestion.loader import PolicyDocument


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClauseSubItem:
    """
    A lettered sub-item within a clause, e.g. ``(a) earnings from employment…``.

    Sub-items belong to their parent clause and are not independent clauses.
    """
    identifier: str   # e.g. "a", "b", "c"
    text: str         # raw text of the item, including the ``(a)`` prefix


@dataclass(frozen=True)
class PolicyClause:
    """
    One authoritative clause extracted from the policy manual.

    Attributes
    ----------
    clause_id:
        The numeric clause identifier as it appears in the document,
        e.g. ``"4.3.2"``.  Never prefixed with § here; that is the
        citation layer's responsibility.
    part_id:
        The Part number extracted from the Part heading, e.g. ``"1"``.
    part_title:
        Full Part heading text, e.g. ``"Part 1 — Scope and Definitions"``.
    section_id:
        The section number, e.g. ``"4.3"``.
    section_title:
        Full section heading text, e.g. ``"4.3 Recipient obligations"``.
    text:
        The complete raw clause text, including the bold opener, any inline
        Markdown, lettered sub-items, and any embedded table rows.
        This is extracted verbatim from the source and must not be modified.
    sub_items:
        Lettered sub-items ``(a)``, ``(b)``, … found within the clause,
        in document order.  Empty tuple when the clause contains none.
    cross_references:
        Distinct ``§``-prefixed references found in the clause text,
        in first-occurrence order.  E.g. ``("§8.5", "§4.3.2")``.
    source_path:
        Absolute path to the source file.
    start_line:
        1-indexed line number of the ``**N.N.N**`` opener in the source.
    end_line:
        1-indexed line number of the last non-empty line of this clause
        before the next clause begins (or end of section).
    effective_from:
        The date from which this clause version is in effect, or ``None``
        if the clause has been in effect since the beginning of the
        consolidated text (i.e. no known effective start date is required
        for original corpus clauses).  Set to the amendment effective date
        for clauses introduced or replaced by an amendment.
    effective_to:
        The last date on which this clause version is in effect, or ``None``
        if the clause has not yet been superseded (open-ended).  Set to the
        day before the replacement amendment's effective date for clauses
        that are later superseded.
    source_document:
        Human-readable identifier of the source document from which this
        clause was extracted.  Defaults to ``"policy_manual.md"`` for the
        original consolidated manual.  Set to the amendment filename for
        clauses introduced or replaced by an amendment.
    """
    # --- Core clause identity and content (existing fields — unchanged) ---
    clause_id: str
    part_id: str
    part_title: str
    section_id: str
    section_title: str
    text: str
    sub_items: tuple[ClauseSubItem, ...]
    cross_references: tuple[str, ...]
    source_path: Path
    start_line: int
    end_line: int
    # --- Temporal metadata (new — all fields have backward-compatible defaults) ---
    effective_from: date | None = None
    effective_to: date | None = None
    source_document: str = "policy_manual.md"


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

# Bold clause opener at the start of a line: **4.3.2**
_RE_CLAUSE_OPENER = re.compile(r'^\*\*(\d+\.\d+\.\d+)\*\*')

# H1 heading
_RE_H1 = re.compile(r'^# (.+)$')

# H2 heading: ## N.N <title>  — we want the section number and full title
_RE_H2 = re.compile(r'^## (\d+\.\d+) (.+)$')

# Part heading: "Part N — …" inside an H1
_RE_PART_NUM = re.compile(r'^Part (\d+)')

# Lettered sub-item: (a), (b), … at start of a (possibly indented) line
_RE_SUB_ITEM = re.compile(r'^\s*\(([a-z])\)\s+(.+)$')

# Cross-reference: §N.N or §N.N.N
_RE_CROSS_REF = re.compile(r'§(\d+\.\d+(?:\.\d+)?)')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_sub_items(lines: list[str]) -> tuple[ClauseSubItem, ...]:
    """Return lettered sub-items found in *lines*, in document order."""
    items: list[ClauseSubItem] = []
    for line in lines:
        m = _RE_SUB_ITEM.match(line)
        if m:
            items.append(ClauseSubItem(identifier=m.group(1), text=line.strip()))
    return tuple(items)


def _extract_cross_refs(text: str) -> tuple[str, ...]:
    """Return deduplicated ``§``-prefixed cross-references in first-occurrence order."""
    seen: dict[str, None] = {}
    for m in _RE_CROSS_REF.finditer(text):
        ref = f"§{m.group(1)}"
        seen[ref] = None
    return tuple(seen.keys())


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------

def _parse_lines(
    lines: list[str],
    source_path: Path,
) -> list[PolicyClause]:
    """
    Walk *lines* and return every authoritative clause found.

    State machine:
    - Track current Part and Section from headings.
    - When a ``**N.N.N**`` opener is found, flush the previous clause
      accumulator and start a new one.
    - All other content (body text, sub-items, table rows) is accumulated
      into the current clause's line buffer.
    """
    clauses: list[PolicyClause] = []

    # Heading context
    current_part_id: str = ""
    current_part_title: str = ""
    current_section_id: str = ""
    current_section_title: str = ""

    # Accumulator for the clause currently being built
    current_clause_id: str = ""
    current_start_line: int = 0
    clause_lines: list[str] = []

    def _flush(end_line: int) -> None:
        """Commit the current accumulator to the clauses list."""
        nonlocal current_clause_id, current_start_line, clause_lines
        if not current_clause_id:
            return
        # Find last non-empty line for end_line
        last_content = end_line
        for i in range(len(clause_lines) - 1, -1, -1):
            if clause_lines[i].strip():
                last_content = current_start_line + i
                break
        raw_text = "\n".join(clause_lines).strip()
        sub_items = _extract_sub_items(clause_lines)
        cross_refs = _extract_cross_refs(raw_text)
        clauses.append(
            PolicyClause(
                clause_id=current_clause_id,
                part_id=current_part_id,
                part_title=current_part_title,
                section_id=current_section_id,
                section_title=current_section_title,
                text=raw_text,
                sub_items=sub_items,
                cross_references=cross_refs,
                source_path=source_path,
                start_line=current_start_line,
                end_line=last_content,
            )
        )
        current_clause_id = ""
        current_start_line = 0
        clause_lines = []

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.rstrip()

        # --- H1: update Part context; flush any open clause ---
        h1 = _RE_H1.match(stripped)
        if h1:
            heading_text = h1.group(1).strip()
            m_part = _RE_PART_NUM.match(heading_text)
            if m_part:
                _flush(line_idx - 1)
                current_part_id = m_part.group(1)
                current_part_title = heading_text
                current_section_id = ""
                current_section_title = ""
            # Non-Part H1 lines (title block) are ignored for clause context
            continue

        # --- H2: update Section context; flush any open clause ---
        h2 = _RE_H2.match(stripped)
        if h2:
            _flush(line_idx - 1)
            current_section_id = h2.group(1)
            current_section_title = f"{h2.group(1)} {h2.group(2).strip()}"
            continue

        # --- Authoritative clause opener ---
        m_clause = _RE_CLAUSE_OPENER.match(stripped)
        if m_clause:
            _flush(line_idx - 1)
            current_clause_id = m_clause.group(1)
            current_start_line = line_idx
            clause_lines = [stripped]
            continue

        # --- Body content: accumulate into current clause if open ---
        if current_clause_id:
            clause_lines.append(stripped)

    # Flush the final clause
    _flush(len(lines))

    return clauses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_clauses(document: PolicyDocument) -> list[PolicyClause]:
    """
    Parse all authoritative clauses from *document*.

    Parameters
    ----------
    document:
        A loaded ``PolicyDocument``.  The ``raw_text`` is not modified.

    Returns
    -------
    list[PolicyClause]
        Clauses in document order.  Each clause has its Part/Section
        context, full source text, sub-items, and cross-references.
    """
    lines = document.raw_text.splitlines()
    return _parse_lines(lines, document.source_path)
