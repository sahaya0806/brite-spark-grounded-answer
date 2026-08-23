"""
Markdown policy document loader.

Responsibilities:
- Accept a path to the policy manual (.md file).
- Validate that the path exists and is a file.
- Read the file as UTF-8 text.
- Preserve the exact raw Markdown content — no normalisation, no modification.
- Return a PolicyDocument dataclass.
- Raise clear exceptions on every failure path.

This module does NOT parse clauses, extract structure, or transform the text.
That is the responsibility of later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PolicyLoadError(Exception):
    """Raised when the policy document cannot be loaded."""


@dataclass(frozen=True)
class PolicyDocument:
    """
    A loaded policy document.

    Attributes
    ----------
    source_path:
        Absolute path to the source file that was loaded.
    raw_text:
        The exact UTF-8 text of the file, byte-for-byte as read.
        This must never be modified after loading.
    character_count:
        Total number of characters in raw_text (len).
    line_count:
        Number of lines produced by str.splitlines().
        Note: this may differ by 1 from the number of newline characters
        depending on whether the file ends with a trailing newline.
    """

    source_path: Path
    raw_text: str
    character_count: int
    line_count: int


def load_policy_document(path: str | Path) -> PolicyDocument:
    """
    Load a Markdown policy document from *path*.

    Parameters
    ----------
    path:
        Path to the ``.md`` file.  May be relative (resolved against the
        current working directory) or absolute.

    Returns
    -------
    PolicyDocument
        An immutable record containing the source path, raw text, and
        basic metrics.

    Raises
    ------
    PolicyLoadError
        If the path does not exist, is not a file, cannot be read as
        UTF-8, or results in an empty document.
    """
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise PolicyLoadError(
            f"Policy document not found: {path!r}"
        )

    if not resolved.is_file():
        raise PolicyLoadError(
            f"Path is not a file: {path!r}"
        )

    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyLoadError(
            f"Policy document is not valid UTF-8: {path!r}"
        ) from exc
    except OSError as exc:
        raise PolicyLoadError(
            f"Could not read policy document: {path!r} — {exc}"
        ) from exc

    if not raw_text.strip():
        raise PolicyLoadError(
            f"Policy document is empty: {path!r}"
        )

    return PolicyDocument(
        source_path=resolved,
        raw_text=raw_text,
        character_count=len(raw_text),
        line_count=len(raw_text.splitlines()),
    )
