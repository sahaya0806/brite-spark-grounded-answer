"""
Citation rendering, extraction, and deterministic validation.

Every citation in a final response must be verifiable against an authoritative
PolicyClause from the parsed corpus.
"""

from __future__ import annotations

import re
from src.ingestion.parser import PolicyClause


# Pattern matching §N.N, §N.N.N, or §N.N.NA (with optional surrounding brackets)
_RE_CITATION = re.compile(r'\[?§(\d+\.\d+(?:\.\d+)?[A-Z]?)\]?')

# Pattern matching "clause N.N.N" or "clause N.N.NA"
_RE_CLAUSE_WORD = re.compile(r'\bclause\s+(\d+\.\d+(?:\.\d+)?[A-Z]?)\b', re.IGNORECASE)


def format_clause_citation(clause: PolicyClause) -> str:
    """
    Format a full verifiable citation for a PolicyClause including source line numbers
    and amendment provenance if applicable.

    Example: "§4.3.2, line 200" or "§6.4.1, line 14 (Amendment No. 2026-01)"
    """
    if clause.start_line == clause.end_line:
        line_part = f"line {clause.start_line}"
    else:
        line_part = f"lines {clause.start_line}–{clause.end_line}"

    amendment_part = ""
    if clause.source_document != "policy_manual.md":
        doc_label = clause.source_document.removesuffix(".md")
        amendment_part = f" ({doc_label})"

    return f"§{clause.clause_id}, {line_part}{amendment_part}"


def format_short_citation(clause: PolicyClause | str) -> str:
    """Format a short citation tag, e.g. '[§4.3.2]'."""
    cid = clause.clause_id if isinstance(clause, PolicyClause) else clause.lstrip("§")
    return f"[§{cid}]"


def extract_cited_clause_ids(text: str) -> list[str]:
    """
    Extract all clause IDs referenced in text, in order of appearance without duplicates.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    # Find §N.N.N patterns
    for m in _RE_CITATION.finditer(text):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    # Find "clause N.N.N" patterns
    for m in _RE_CLAUSE_WORD.finditer(text):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    return ordered


def validate_citations(
    cited_ids: list[str],
    allowed_ids: set[str] | frozenset[str],
) -> list[str]:
    """
    Filter extracted clause IDs to only those in the allowed set.
    Any hallucinated or unsupplied clause ID is rejected.
    """
    return [cid for cid in cited_ids if cid in allowed_ids]


def sanitize_text_citations(
    text: str,
    allowed_ids: set[str] | frozenset[str],
) -> str:
    """
    Remove any hallucinated clause citations from text that are not in allowed_ids.
    """
    def _replace(match: re.Match) -> str:
        cid = match.group(1)
        if cid in allowed_ids:
            return match.group(0)  # Keep valid citation
        return ""  # Strip hallucinated citation

    sanitized = _RE_CITATION.sub(_replace, text)
    # Clean up double spaces or orphaned brackets created by removal
    sanitized = re.sub(r'\[\s*\]', '', sanitized)
    sanitized = re.sub(r'  +', ' ', sanitized).strip()
    return sanitized
