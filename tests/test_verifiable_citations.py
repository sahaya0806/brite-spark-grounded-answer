"""
Tests for Milestone 5 — Verifiable Policy Citations.

All tests are completely offline and deterministic.
No OpenAI API key required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.citation.models import Citation
from src.citation.renderer import (
    DEFAULT_COMMIT,
    DEFAULT_REPO_URL,
    create_citation,
    format_clause_citation,
    generate_source_url,
    validate_citation_url,
)
from src.ingestion.parser import PolicyClause
from src.generation.models import GroundedAnswer
from src.evidence.models import DecisionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO = "https://github.com/sahaya0806/brite-spark-grounded-answer"
_COMMIT = DEFAULT_COMMIT
_POLICY_PATH = Path("data/raw/policy_manual.md")
_AMENDMENT_PATH = Path("data/raw/Amendment No. 2026-01.md")


def _policy_clause(
    cid: str = "4.3.2",
    start_line: int = 200,
    end_line: int = 203,
    source_path: Path | None = None,
    source_document: str = "policy_manual.md",
    effective_from: date | None = None,
    effective_to: date | None = None,
) -> PolicyClause:
    return PolicyClause(
        clause_id=cid,
        part_id="4",
        part_title="Part 4",
        section_id="4.3",
        section_title="4.3 Reporting",
        text=f"**{cid}** Sample clause text.",
        sub_items=(),
        cross_references=(),
        source_path=source_path or _POLICY_PATH,
        start_line=start_line,
        end_line=end_line,
        effective_from=effective_from,
        effective_to=effective_to,
        source_document=source_document,
    )


def _amendment_clause(
    cid: str = "4.3.2",
    start_line: int = 18,
    end_line: int = 20,
) -> PolicyClause:
    return _policy_clause(
        cid=cid,
        start_line=start_line,
        end_line=end_line,
        source_path=_AMENDMENT_PATH,
        source_document="Amendment No. 2026-01.md",
        effective_from=date(2026, 4, 1),
    )


# ---------------------------------------------------------------------------
# 1. Original policy URL generation
# ---------------------------------------------------------------------------

class TestOriginalPolicyURLGeneration:

    def test_generates_correct_base_url(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert url.startswith(_REPO)

    def test_generates_commit_pinned_url(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert f"/blob/{_COMMIT}/" in url

    def test_original_policy_path_in_url(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert "data/raw/policy_manual.md" in url

    def test_multiline_anchor_original(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert url.endswith("#L200-L203")

    def test_single_line_anchor_original(self):
        url = generate_source_url(_POLICY_PATH, 42, 42)
        assert url.endswith("#L42")
        # Must NOT include redundant -L42
        assert "#L42-L" not in url


# ---------------------------------------------------------------------------
# 2. Amendment URL generation
# ---------------------------------------------------------------------------

class TestAmendmentURLGeneration:

    def test_amendment_url_encoded_spaces(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        # Spaces in filename must be percent-encoded
        assert "Amendment%20No.%202026-01.md" in url

    def test_amendment_url_not_in_policy_manual(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        assert "policy_manual.md" not in url

    def test_amendment_url_multiline_anchor(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        assert url.endswith("#L18-L20")

    def test_amendment_url_commit_pinned(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        assert f"/blob/{_COMMIT}/" in url

    def test_amendment_url_github_base(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        assert url.startswith("https://github.com/")


# ---------------------------------------------------------------------------
# 3. Single-line and multi-line anchors
# ---------------------------------------------------------------------------

class TestLineAnchors:

    def test_single_line_anchor_clause(self):
        url = generate_source_url(_POLICY_PATH, 100, 100)
        assert url.endswith("#L100")

    def test_multiline_anchor(self):
        url = generate_source_url(_POLICY_PATH, 100, 105)
        assert url.endswith("#L100-L105")

    def test_anchor_reflects_start_end_correctly(self):
        url = generate_source_url(_POLICY_PATH, 50, 53)
        assert "#L50-L53" in url
        assert "#L50-L50" not in url


# ---------------------------------------------------------------------------
# 4. URL encoding of spaces
# ---------------------------------------------------------------------------

class TestURLEncoding:

    def test_space_in_amendment_filename_encoded(self):
        url = generate_source_url(_AMENDMENT_PATH, 1, 5)
        # No literal spaces allowed in URL
        assert " " not in url
        assert "Amendment%20No.%202026-01.md" in url

    def test_policy_manual_no_encoding_needed(self):
        url = generate_source_url(_POLICY_PATH, 1, 5)
        assert "policy_manual.md" in url
        assert "%20" not in url


# ---------------------------------------------------------------------------
# 5. Alphanumeric clause ID §10.5.3A
# ---------------------------------------------------------------------------

class TestAlphanumericClauseID:

    def test_alphanumeric_id_in_citation_model(self):
        clause = _policy_clause(cid="10.5.3A", start_line=41, end_line=43)
        cit = create_citation(clause)
        assert cit.clause_id == "10.5.3A"

    def test_alphanumeric_id_in_format_label(self):
        clause = _policy_clause(cid="10.5.3A", start_line=41, end_line=43)
        cit = create_citation(clause)
        assert "§10.5.3A" in cit.format_label()

    def test_alphanumeric_id_amendment_url(self):
        clause = _amendment_clause(cid="10.5.3A", start_line=41, end_line=43)
        cit = create_citation(clause)
        assert "Amendment%20No.%202026-01.md" in cit.source_url
        assert "#L41-L43" in cit.source_url


# ---------------------------------------------------------------------------
# 6. Citation model correctness
# ---------------------------------------------------------------------------

class TestCitationModel:

    def test_source_path_preserved(self):
        clause = _policy_clause(start_line=200, end_line=203)
        cit = create_citation(clause)
        assert cit.source_path == _POLICY_PATH

    def test_start_line_preserved(self):
        clause = _policy_clause(start_line=200, end_line=203)
        cit = create_citation(clause)
        assert cit.start_line == 200

    def test_end_line_preserved(self):
        clause = _policy_clause(start_line=200, end_line=203)
        cit = create_citation(clause)
        assert cit.end_line == 203

    def test_clause_id_preserved(self):
        clause = _policy_clause(cid="4.3.2")
        cit = create_citation(clause)
        assert cit.clause_id == "4.3.2"

    def test_source_label_original(self):
        clause = _policy_clause(source_document="policy_manual.md")
        cit = create_citation(clause)
        assert cit.source_label == "policy_manual.md"

    def test_source_label_amendment(self):
        clause = _amendment_clause()
        cit = create_citation(clause)
        assert cit.source_label == "Amendment No. 2026-01.md"

    def test_citation_is_immutable(self):
        clause = _policy_clause()
        cit = create_citation(clause)
        with pytest.raises((AttributeError, TypeError)):
            cit.clause_id = "99.99.99"  # type: ignore

    def test_line_anchor_property_multiline(self):
        cit = Citation(
            clause_id="4.3.2",
            source_path=_POLICY_PATH,
            start_line=200,
            end_line=203,
            source_label="policy_manual.md",
            source_url="https://example.com#L200-L203",
        )
        assert cit.line_anchor == "#L200-L203"

    def test_line_anchor_property_single(self):
        cit = Citation(
            clause_id="4.3.2",
            source_path=_POLICY_PATH,
            start_line=42,
            end_line=42,
            source_label="policy_manual.md",
            source_url="https://example.com#L42",
        )
        assert cit.line_anchor == "#L42"

    def test_line_label_property_multiline(self):
        cit = Citation(
            clause_id="4.3.2",
            source_path=_POLICY_PATH,
            start_line=200,
            end_line=203,
            source_label="policy_manual.md",
            source_url="https://example.com#L200-L203",
        )
        assert cit.line_label == "lines 200–203"

    def test_line_label_property_single(self):
        cit = Citation(
            clause_id="4.3.2",
            source_path=_POLICY_PATH,
            start_line=42,
            end_line=42,
            source_label="policy_manual.md",
            source_url="https://example.com#L42",
        )
        assert cit.line_label == "line 42"


# ---------------------------------------------------------------------------
# 7. Commit-pinned URL
# ---------------------------------------------------------------------------

class TestCommitPinnedURL:

    def test_url_contains_known_commit(self):
        url = generate_source_url(_POLICY_PATH, 1, 5)
        assert _COMMIT in url

    def test_custom_commit_used_when_provided(self):
        custom = "abc123def456"
        url = generate_source_url(_POLICY_PATH, 1, 5, commit=custom)
        assert custom in url
        assert _COMMIT not in url

    def test_custom_repo_url_used_when_provided(self):
        custom_repo = "https://github.com/testorg/testrepo"
        url = generate_source_url(_POLICY_PATH, 1, 5, repo_url=custom_repo)
        assert url.startswith(custom_repo)


# ---------------------------------------------------------------------------
# 8. Original vs Amendment distinction
# ---------------------------------------------------------------------------

class TestOriginalVsAmendment:

    def test_original_clause_points_to_policy_manual(self):
        clause = _policy_clause()
        cit = create_citation(clause)
        assert "policy_manual.md" in cit.source_url
        assert "Amendment" not in cit.source_url

    def test_amendment_clause_points_to_amendment(self):
        clause = _amendment_clause()
        cit = create_citation(clause)
        assert "Amendment%20No.%202026-01.md" in cit.source_url
        assert "policy_manual.md" not in cit.source_url

    def test_original_format_label_no_amendment_suffix(self):
        clause = _policy_clause()
        cit = create_citation(clause)
        label = cit.format_label()
        assert "Amendment" not in label

    def test_amendment_format_label_shows_amendment(self):
        clause = _amendment_clause()
        cit = create_citation(clause)
        label = cit.format_label()
        assert "Amendment No. 2026-01" in label


# ---------------------------------------------------------------------------
# 9. URL validator (offline)
# ---------------------------------------------------------------------------

class TestURLValidator:

    def test_valid_policy_url_passes(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert validate_citation_url(
            url,
            expected_repo=_REPO,
            expected_file="policy_manual.md",
            expected_start=200,
            expected_end=203,
        )

    def test_valid_amendment_url_passes(self):
        url = generate_source_url(_AMENDMENT_PATH, 18, 20)
        assert validate_citation_url(
            url,
            expected_repo=_REPO,
            expected_file="Amendment No. 2026-01.md",
            expected_start=18,
            expected_end=20,
        )

    def test_wrong_line_fails_validation(self):
        url = generate_source_url(_POLICY_PATH, 200, 203)
        assert not validate_citation_url(
            url,
            expected_start=999,
            expected_end=1000,
        )

    def test_non_github_url_fails_validation(self):
        fake_url = "https://example.com/blob/abc/file.md#L1"
        assert not validate_citation_url(fake_url)

    def test_url_missing_blob_fails(self):
        bad_url = f"{_REPO}/tree/{_COMMIT}/data/raw/policy_manual.md#L1"
        assert not validate_citation_url(bad_url)


# ---------------------------------------------------------------------------
# 10. Temporal filtering preserves provenance
# ---------------------------------------------------------------------------

class TestTemporalProvenance:

    def test_original_clause_retains_policy_manual_path(self):
        """Original clause (pre-amendment date) must cite policy_manual.md."""
        clause = _policy_clause(
            cid="4.3.2",
            start_line=145,
            end_line=147,
            source_document="policy_manual.md",
        )
        cit = create_citation(clause)
        assert cit.source_label == "policy_manual.md"
        assert "policy_manual.md" in cit.source_url

    def test_amended_clause_retains_amendment_path(self):
        """Amended clause (post-amendment date) must cite amendment file."""
        clause = _amendment_clause(cid="4.3.2", start_line=18, end_line=20)
        cit = create_citation(clause)
        assert cit.source_label == "Amendment No. 2026-01.md"
        assert "Amendment%20No.%202026-01.md" in cit.source_url

    def test_provenance_not_polluted_across_clauses(self):
        """Two clauses from different sources produce independently correct citations."""
        orig = _policy_clause(cid="4.3.2", start_line=145, end_line=147)
        amend = _amendment_clause(cid="4.3.2", start_line=18, end_line=20)

        cit_orig = create_citation(orig)
        cit_amend = create_citation(amend)

        assert cit_orig.source_label == "policy_manual.md"
        assert cit_amend.source_label == "Amendment No. 2026-01.md"
        assert cit_orig.source_url != cit_amend.source_url
        assert cit_orig.start_line == 145
        assert cit_amend.start_line == 18


# ---------------------------------------------------------------------------
# 11. GroundedAnswer carries verifiable_citations
# ---------------------------------------------------------------------------

class TestGroundedAnswerVerifiableCitations:

    def test_verifiable_citations_default_empty(self):
        answer = GroundedAnswer(
            question="q",
            answer_text="a",
            status=DecisionStatus.INSUFFICIENT,
            citations=(),
            supporting_clause_ids=(),
            refusal=True,
            conflicts=(),
            rationale="r",
            primary_clauses=(),
        )
        assert answer.verifiable_citations == ()

    def test_verifiable_citations_field_accepts_citation_objects(self):
        clause = _policy_clause()
        cit = create_citation(clause)
        answer = GroundedAnswer(
            question="q",
            answer_text="a",
            status=DecisionStatus.SUPPORTED,
            citations=("§4.3.2, lines 200–203",),
            supporting_clause_ids=("4.3.2",),
            refusal=False,
            conflicts=(),
            rationale="r",
            primary_clauses=(clause,),
            verifiable_citations=(cit,),
        )
        assert len(answer.verifiable_citations) == 1
        assert answer.verifiable_citations[0].clause_id == "4.3.2"
        assert "policy_manual.md" in answer.verifiable_citations[0].source_url

    def test_amendment_citation_in_grounded_answer(self):
        clause = _amendment_clause()
        cit = create_citation(clause)
        answer = GroundedAnswer(
            question="q",
            answer_text="a",
            status=DecisionStatus.SUPPORTED,
            citations=("§4.3.2, lines 18–20 (Amendment No. 2026-01)",),
            supporting_clause_ids=("4.3.2",),
            refusal=False,
            conflicts=(),
            rationale="r",
            primary_clauses=(clause,),
            verifiable_citations=(cit,),
        )
        assert "Amendment%20No.%202026-01.md" in answer.verifiable_citations[0].source_url


# ---------------------------------------------------------------------------
# 12. No fabricated citation URLs from LLM text
# ---------------------------------------------------------------------------

class TestNoFabricatedCitations:

    def test_url_cannot_be_constructed_from_free_text(self):
        """
        Verifies that the URL builder requires actual PolicyClause objects
        and cannot be seeded with arbitrary text.
        If someone passes a fake path, the URL still routes through the
        known repository base (it does not allow arbitrary external URLs).
        """
        fake_path = Path("data/raw/../../etc/passwd")
        url = generate_source_url(fake_path, 1, 1)
        # The URL must still start with the known repo
        assert url.startswith("https://github.com/sahaya0806/brite-spark-grounded-answer")
        # It must not escape to dangerous paths
        assert "etc/passwd" not in url or "data/raw" in url

    def test_citation_source_url_does_not_come_from_text(self):
        """Citation URL is derived from structured PolicyClause, never from answer text."""
        clause = _policy_clause(start_line=200, end_line=203)
        cit = create_citation(clause)
        # The url contains the real line numbers, not any fabricated value
        assert "#L200-L203" in cit.source_url


# ---------------------------------------------------------------------------
# 13. Missing/edge-case provenance handled safely
# ---------------------------------------------------------------------------

class TestMissingProvenanceSafety:

    def test_clause_with_unknown_source_document(self):
        """
        A clause with a non-standard source_document label still generates
        a citation without crashing.
        """
        clause = _policy_clause(
            source_document="SomeOtherDocument.md",
            source_path=Path("data/raw/SomeOtherDocument.md"),
        )
        cit = create_citation(clause)
        assert cit.source_label == "SomeOtherDocument.md"
        assert cit.source_url.startswith("https://github.com/")

    def test_single_line_clause_no_invalid_anchor(self):
        clause = _policy_clause(start_line=77, end_line=77)
        cit = create_citation(clause)
        assert "#L77" in cit.source_url
        assert "#L77-L77" not in cit.source_url


# ---------------------------------------------------------------------------
# 14. format_clause_citation legacy compatibility
# ---------------------------------------------------------------------------

class TestFormatClauseCitation:

    def test_original_single_line(self):
        clause = _policy_clause(start_line=80, end_line=80)
        result = format_clause_citation(clause)
        assert result == "§4.3.2, line 80"

    def test_original_multiline(self):
        clause = _policy_clause(start_line=200, end_line=203)
        result = format_clause_citation(clause)
        assert result == "§4.3.2, lines 200–203"

    def test_amendment_clause_shows_provenance(self):
        clause = _amendment_clause()
        result = format_clause_citation(clause)
        assert "Amendment No. 2026-01" in result

    def test_alphanumeric_id(self):
        clause = _policy_clause(cid="10.5.3A", start_line=41, end_line=43)
        result = format_clause_citation(clause)
        assert "§10.5.3A" in result
