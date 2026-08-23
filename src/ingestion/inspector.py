"""
Lightweight Markdown structural inspector.

Responsibilities:
- Accept a PolicyDocument.
- Report structural observations about the Markdown: headings, lists,
  tables, possible clause identifiers, cross-reference patterns.
- Return an immutable MarkdownInspection dataclass.

This module does NOT create authoritative clause objects.
It does NOT modify the document.
It is deliberately separated from the loader so that structural observations
can be made, reviewed, and used to inform the Milestone 3 clause parser design
without coupling inspection concerns to the loading concerns.

All patterns are based on the actual structure of:
  data/raw/policy_manual.md  (Calder County Household Support Program)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.ingestion.loader import PolicyDocument


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadingInfo:
    """A single heading extracted from the document."""

    level: int          # 1 = H1, 2 = H2, etc.
    text: str           # Raw heading text after the leading '#' characters
    line_number: int    # 1-indexed line number in the source


@dataclass(frozen=True)
class MarkdownInspection:
    """
    Structural observations about a loaded Markdown policy document.

    These are OBSERVATIONS only — they are inputs to the clause parser
    design (Milestone 3), not the final clause representation.

    Attributes
    ----------
    total_lines:
        Line count (mirrors PolicyDocument.line_count).
    total_characters:
        Character count (mirrors PolicyDocument.character_count).
    headings:
        Every heading found, in document order.
    heading_counts_by_level:
        Mapping from heading level (int) to count.
    table_row_count:
        Number of Markdown table rows detected (lines starting with ``|``).
    ordered_list_item_count:
        Lines matching a lettered sub-list item ``(a)``, ``(b)``, etc.
        This is the primary list style in the corpus.
    unordered_list_item_count:
        Lines matching ``-``, ``*``, or ``+`` bullet items.
    possible_clause_ids:
        Distinct bold paragraph-opener identifiers of the form
        ``N.N.N`` (e.g. ``**4.3.2**``).  These are candidates for
        clause IDs in the Milestone 3 parser.  Not yet authoritative.
    cross_reference_patterns:
        Distinct §-prefixed references found in the text
        (e.g. ``§4.3.2``, ``§6.4``).
    """

    total_lines: int
    total_characters: int
    headings: tuple[HeadingInfo, ...]
    heading_counts_by_level: dict[int, int]
    table_row_count: int
    ordered_list_item_count: int
    unordered_list_item_count: int
    possible_clause_ids: tuple[str, ...]
    cross_reference_patterns: tuple[str, ...]


# ---------------------------------------------------------------------------
# Regex patterns  (compiled once at import time)
# ---------------------------------------------------------------------------

# Heading: one or more '#' followed by a space
_RE_HEADING = re.compile(r'^(#{1,6}) (.+)$')

# Table row: line starting with '|'
_RE_TABLE_ROW = re.compile(r'^\s*\|')

# Lettered sub-list item: (a), (b), …  — the primary list style in this corpus
_RE_LETTERED_ITEM = re.compile(r'^\s*\([a-z]\)\s')

# Standard Markdown unordered list item
_RE_UNORDERED_ITEM = re.compile(r'^\s*[-*+] ')

# Bold clause-number paragraph openers:  **4.3.2**  at the start of a line
# Captures the numeric identifier only (without the ** markers).
_RE_BOLD_CLAUSE_ID = re.compile(r'^\*\*(\d+\.\d+(?:\.\d+)?)\*\*', re.MULTILINE)

# Cross-references of the form §4.3.2 or §6.4
_RE_CROSS_REF = re.compile(r'§(\d+\.\d+(?:\.\d+)?)')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_markdown(document: PolicyDocument) -> MarkdownInspection:
    """
    Inspect the structural properties of *document*.

    Parameters
    ----------
    document:
        A loaded PolicyDocument (from ``load_policy_document``).

    Returns
    -------
    MarkdownInspection
        Immutable structural observations.  The document's ``raw_text``
        is not modified.
    """
    raw = document.raw_text
    lines = raw.splitlines()

    headings: list[HeadingInfo] = []
    table_rows = 0
    lettered_items = 0
    unordered_items = 0

    for line_number, line in enumerate(lines, start=1):
        # --- headings ---
        m = _RE_HEADING.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            headings.append(HeadingInfo(level=level, text=text, line_number=line_number))
            continue  # a heading line cannot be anything else

        # --- table rows ---
        if _RE_TABLE_ROW.match(line):
            table_rows += 1
            continue

        # --- lettered list items ---
        if _RE_LETTERED_ITEM.match(line):
            lettered_items += 1
            continue

        # --- unordered list items ---
        if _RE_UNORDERED_ITEM.match(line):
            unordered_items += 1

    # heading counts by level
    counts_by_level: dict[int, int] = {}
    for h in headings:
        counts_by_level[h.level] = counts_by_level.get(h.level, 0) + 1

    # possible clause IDs from the full text (preserve order, deduplicate)
    seen_ids: dict[str, None] = {}
    for m in _RE_BOLD_CLAUSE_ID.finditer(raw):
        seen_ids[m.group(1)] = None
    possible_clause_ids = tuple(seen_ids.keys())

    # cross-reference patterns (preserve order, deduplicate)
    seen_xrefs: dict[str, None] = {}
    for m in _RE_CROSS_REF.finditer(raw):
        seen_xrefs[m.group(1)] = None
    cross_reference_patterns = tuple(seen_xrefs.keys())

    return MarkdownInspection(
        total_lines=document.line_count,
        total_characters=document.character_count,
        headings=tuple(headings),
        heading_counts_by_level=counts_by_level,
        table_row_count=table_rows,
        ordered_list_item_count=lettered_items,
        unordered_list_item_count=unordered_items,
        possible_clause_ids=possible_clause_ids,
        cross_reference_patterns=cross_reference_patterns,
    )
