"""
Verifiable Citation data model.

Represents an authoritative, immutable citation pointing directly to exact source
lines in the policy repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Citation:
    """
    Structured, verifiable policy citation.

    Attributes
    ----------
    clause_id:
        Policy clause identifier (e.g. "4.3.2" or "10.5.3A").
    source_path:
        Path to the source markdown file (e.g. Path("data/raw/policy_manual.md")).
    start_line:
        1-based starting line number in the source file.
    end_line:
        1-based ending line number in the source file.
    source_label:
        Human-readable document label (e.g. "policy_manual.md" or "Amendment No. 2026-01").
    source_url:
        Commit-pinned GitHub URL pointing to the exact source lines.
    """

    clause_id: str
    source_path: Path
    start_line: int
    end_line: int
    source_label: str
    source_url: str

    @property
    def line_anchor(self) -> str:
        """GitHub line anchor fragment (e.g. '#L200' or '#L200-L203')."""
        if self.start_line == self.end_line:
            return f"#L{self.start_line}"
        return f"#L{self.start_line}-L{self.end_line}"

    @property
    def line_label(self) -> str:
        """Human-readable line description (e.g. 'line 200' or 'lines 200–203')."""
        if self.start_line == self.end_line:
            return f"line {self.start_line}"
        return f"lines {self.start_line}–{self.end_line}"

    def format_label(self) -> str:
        """
        Format standard citation label.

        Examples
        --------
        - "§4.3.2, line 200"
        - "§4.3.2, lines 18–20 (Amendment No. 2026-01)"
        - "§10.5.3A, lines 41–43 (Amendment No. 2026-01)"
        """
        amendment_suffix = ""
        if self.source_label != "policy_manual.md":
            label = self.source_label.removesuffix(".md")
            amendment_suffix = f" ({label})"
        return f"§{self.clause_id}, {self.line_label}{amendment_suffix}"
