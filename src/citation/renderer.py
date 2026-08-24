"""
Citation rendering, extraction, URL generation, and deterministic validation.

Every citation in a final response must be verifiable against an authoritative
PolicyClause from the parsed corpus with direct links to the exact source lines.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import urllib.parse

from src.citation.models import Citation
from src.ingestion.parser import PolicyClause

DEFAULT_REPO_URL = os.getenv(
    "GITHUB_REPO_URL",
    "https://github.com/sahaya0806/brite-spark-grounded-answer",
)
DEFAULT_COMMIT = os.getenv(
    "GITHUB_COMMIT",
    "0827c39fdaa69274f3da3b11b3fb49bd52d1912f",
)

# Pattern matching §N.N, §N.N.N, or §N.N.NA (with optional surrounding brackets)
_RE_CITATION = re.compile(r'\[?§(\d+\.\d+(?:\.\d+)?[A-Z]?)\]?')

# Pattern matching "clause N.N.N" or "clause N.N.NA"
_RE_CLAUSE_WORD = re.compile(r'\bclause\s+(\d+\.\d+(?:\.\d+)?[A-Z]?)\b', re.IGNORECASE)


def generate_source_url(
    source_path: Path | str,
    start_line: int,
    end_line: int,
    repo_url: str | None = None,
    commit: str | None = None,
) -> str:
    """
    Generate a commit-pinned GitHub URL pointing to the exact line range anchor.

    Parameters
    ----------
    source_path:
        Path to the source markdown document.
    start_line:
        1-based starting line number.
    end_line:
        1-based ending line number.
    repo_url:
        Base GitHub repository URL (defaults to DEFAULT_REPO_URL).
    commit:
        Commit SHA for pinning (defaults to DEFAULT_COMMIT).

    Returns
    -------
    str
        Full GitHub blob URL with #L anchor, e.g.:
        "https://github.com/sahaya0806/brite-spark-grounded-answer/blob/0827c39.../data/raw/Amendment%20No.%202026-01.md#L18-L20"
    """
    base_url = (repo_url or DEFAULT_REPO_URL).rstrip("/")
    commit_sha = commit or DEFAULT_COMMIT

    p = Path(source_path)
    # Normalize relative path within repository
    parts = p.parts
    if "data" in parts:
        data_idx = parts.index("data")
        rel_parts = parts[data_idx:]
    else:
        rel_parts = ("data", "raw", p.name)

    # URL encode path segments (e.g. spaces in "Amendment No. 2026-01.md" -> "%20")
    encoded_path = "/".join(urllib.parse.quote(part) for part in rel_parts)

    if start_line == end_line:
        anchor = f"#L{start_line}"
    else:
        anchor = f"#L{start_line}-L{end_line}"

    return f"{base_url}/blob/{commit_sha}/{encoded_path}{anchor}"


def create_citation(
    clause: PolicyClause,
    repo_url: str | None = None,
    commit: str | None = None,
) -> Citation:
    """
    Construct an immutable, verifiable Citation object directly from a PolicyClause.
    """
    source_url = generate_source_url(
        source_path=clause.source_path,
        start_line=clause.start_line,
        end_line=clause.end_line,
        repo_url=repo_url,
        commit=commit,
    )
    return Citation(
        clause_id=clause.clause_id,
        source_path=clause.source_path,
        start_line=clause.start_line,
        end_line=clause.end_line,
        source_label=clause.source_document,
        source_url=source_url,
    )


def validate_citation_url(
    url: str,
    expected_repo: str | None = None,
    expected_file: str | None = None,
    expected_start: int | None = None,
    expected_end: int | None = None,
) -> bool:
    """
    Offline deterministic validator ensuring citation URLs are well-formed.
    """
    if not url.startswith("https://github.com/"):
        return False

    repo = expected_repo or DEFAULT_REPO_URL
    if not url.startswith(repo):
        return False

    if "/blob/" not in url:
        return False

    if expected_file:
        encoded_file = urllib.parse.quote(expected_file)
        if encoded_file not in url and expected_file not in url:
            return False

    if expected_start is not None:
        if expected_end is not None and expected_start != expected_end:
            expected_anchor = f"#L{expected_start}-L{expected_end}"
        else:
            expected_anchor = f"#L{expected_start}"

        if not url.endswith(expected_anchor):
            return False

    return True


def format_clause_citation(clause: PolicyClause) -> str:
    """
    Format a full verifiable citation label for a PolicyClause including source line numbers
    and amendment provenance if applicable.

    Example: "§4.3.2, line 200" or "§6.4.1, lines 18–20 (Amendment No. 2026-01)"
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
