# AI Usage Record

This file honestly documents the use of AI assistance during development of
The Grounded Answer. It is updated incrementally as the project progresses.

---

## Tools Used

### Kiro (coding assistant)
- Used for: code generation, file scaffolding, test writing, refactoring (Milestones 1–4, initial Milestone 5 draft).
- All generated code is reviewed by the project team before committing.
- The team is responsible for verifying correctness, running tests, and
  making final implementation decisions.

### Antigravity (coding assistant)
- Used for: completing Milestone 5 implementation, adding edge-case tests,
  refining models and evaluators, and updating architecture documentation.

### ChatGPT
- Used for: architecture planning, reasoning about the RAG pipeline design,
  reviewing the problem statement, and discussing trade-offs.
- Planning output is treated as input to the team's decision process, not as
  a final specification.

---

## Milestone 1 — Project Foundation

- Kiro generated the initial project scaffold: directory structure,
  `__init__.py` files, `requirements.txt`, `.gitignore`, `pytest.ini`,
  the CLI entry point (`src/app.py`), and the smoke test suite.
- DECISIONS.md content was drafted with Kiro assistance and reviewed by
  the team.
- README.md was drafted with Kiro assistance and reviewed by the team.
- The team verified that all smoke tests pass and that the entry point
  runs correctly before committing.

---

## Post-Milestone 1 — Corpus format clarification

- The project team clarified with the Brite Spark organizers that the
  supplied policy corpus is a Markdown (`.md`) file, not a PDF.
- The initial implementation plan assumed PDF ingestion via PyMuPDF.
- Kiro assisted with updating `requirements.txt` (removing `pymupdf`) and
  drafting ADR-005 in DECISIONS.md to record the corrected ingestion approach.
- The team reviewed and approved all changes before committing.
- No parsing code was written yet; this update covers only the plan and
  dependency corrections.

---

## Principles

The project team remains responsible for:
- All architecture decisions (documented in DECISIONS.md).
- Reviewing and understanding every piece of generated code.
- Designing the evaluation set (must not be made artificially easy).
- Debugging failures.
- Final judgement on refusal thresholds and citation strategy.

AI assistance accelerates implementation. It does not replace engineering
judgement.

---

## Milestone 2 — Markdown Policy Ingestion

- Kiro assisted with implementing `src/ingestion/loader.py` (PolicyDocument
  dataclass, load_policy_document function, error handling).
- Kiro assisted with implementing `src/ingestion/inspector.py`
  (MarkdownInspection dataclass, inspect_markdown function, regex patterns).
- Kiro assisted with writing the full test suite in
  `tests/test_ingestion.py` (53 tests covering loader, inspector, and real
  corpus integration).
- Kiro assisted with updating DECISIONS.md (ADR-006, ADR-007, ADR-008),
  AI-USAGE.md, and README.md.
- The project team inspected the real policy corpus
  (`data/raw/policy_manual.md`) before any implementation was written,
  and reviewed all generated code and documentation before committing.
- The project team identified the intentional contradiction (§4.3.2 vs
  §9.1.4 reporting window) and the apparent cross-reference gap (§7.1.3)
  during corpus inspection.
- All 53 tests were verified to pass before committing.

---

## Milestone 3 — Clause-Level Parsing and Structured Clause Store

- Kiro assisted with implementing ``src/ingestion/parser.py``:
  the ``PolicyClause`` and ``ClauseSubItem`` dataclasses, the ``parse_clauses``
  function, and the line-accumulation state machine.
- Kiro assisted with implementing ``src/ingestion/store.py``:
  the ``ClauseStore`` class and ``ClauseNotFoundError``.
- Kiro assisted with writing the test suite in ``tests/test_parser.py``
  (72 tests covering basic extraction, Part/Section association, sub-items,
  cross-references, source line tracking, ClauseStore API, table handling,
  non-inflation guards, determinism, and 24 real corpus integration tests).
- Kiro assisted with updating DECISIONS.md (ADR-009 through ADR-015),
  AI-USAGE.md, and README.md.
- The project team verified parser output against the real corpus before
  writing any tests, confirmed the 137-clause count, and reviewed all
  implementation and documentation before committing.
- The project team confirmed that the intentional contradiction (§4.3.2 vs
  §9.1.4) and the apparent gap (§7.1.3 → §5.4) are correctly preserved in
  the parsed output and are explicitly covered by tests.
- All 125 tests were verified to pass before committing.

---

## Milestone 4 — Hybrid Policy Retrieval

- Kiro assisted with implementing the retrieval package:
  ``src/retrieval/models.py`` (``RetrievalResult``),
  ``src/retrieval/embeddings.py`` (``EmbeddingProvider`` Protocol,
  ``OpenAIEmbeddingProvider``, ``FakeEmbeddingProvider``),
  ``src/retrieval/vector.py`` (``VectorIndex`` using FAISS),
  ``src/retrieval/lexical.py`` (``LexicalIndex`` using BM25Okapi),
  ``src/retrieval/hybrid.py`` (``HybridRetriever``, ``RetrieverConfig``,
  RRF merging).
- Kiro assisted with writing the test suite in ``tests/test_retrieval.py``
  (80 tests across 9 groups covering tokeniser, embedding interface,
  vector index, lexical index, hybrid API, deduplication, edge cases,
  result model, and 20 real corpus integration tests).
- Kiro assisted with updating DECISIONS.md (ADR-016 through ADR-022),
  AI-USAGE.md, README.md, and ``.env.example``.
- The project team verified that all 205 tests pass, reviewed the
  implementation, and confirmed that:
  - No OpenAI API key is required for the test suite.
  - Both contradiction clauses (§4.3.2 and §9.1.4) are reachable for
    reporting-deadline queries.
  - The apparent gap clause (§7.1.3) is retrievable.
  - The retriever does not generate answers or make refusal decisions.

---

## Milestone 5 — Evidence Evaluation and Decision Layer

- Kiro generated the initial draft of the evidence evaluation modules
  (`src/evidence/models.py`, `src/evidence/scoring.py`,
  `src/evidence/contradiction.py`, `src/evidence/evaluator.py`,
  `src/evidence/__init__.py`, and initial tests in `tests/test_evidence.py`).
- Antigravity completed and refined the Milestone 5 implementation:
  - Added input deduplication by `clause_id` and strict provenance validation in `src/evidence/evaluator.py`.
  - Added the `supporting_clause_ids` convenience property in `src/evidence/models.py`.
  - Fixed threshold calibration in `tests/test_evidence.py`.
  - Added unit and edge case tests for multi-clause support, scope/applicability differentiation, deduplication, and hallucinated clause ID prevention (81 tests total in `test_evidence.py`).
  - Drafted ADR-023 through ADR-027 in `DECISIONS.md`.
  - Updated `README.md` with Evidence Evaluation API documentation.
- The project team verified that all 286 tests pass across the entire suite, confirmed that evidence evaluation is strictly separated from retrieval, and verified the three-way decision behavior (`SUPPORTED`, `INSUFFICIENT`, `CONFLICTING`) on both synthetic and real-corpus test cases.

---

## Milestone 6 — Grounded Answer Generation

- Antigravity assisted with implementing the grounded answer generation package:
  - `src/generation/providers.py` (`ChatProvider` Protocol, `OpenAIChatProvider` using `gpt-4o-mini`, and deterministic `FakeChatProvider` for offline testing).
  - `src/citation/renderer.py` and `src/citation/__init__.py` (`format_clause_citation`, `format_short_citation`, `extract_cited_clause_ids`, `validate_citations`, `sanitize_text_citations`).
  - `src/generation/models.py` (`GroundedAnswer`).
  - `src/generation/prompts.py` (`SYSTEM_PROMPT` and `build_grounded_prompt`).
  - `src/generation/generator.py` (`GroundedAnswerGenerator` implementing `SUPPORTED`, `INSUFFICIENT`, and `CONFLICTING` paths).
  - `src/pipeline.py` (`PolicyQAPipeline` unifying retrieval, evidence evaluation, and answer generation).
  - `src/app.py` (CLI `ask` and `info` commands).
- Antigravity assisted with writing comprehensive unit and end-to-end tests:
  - `tests/test_generation.py` (25 tests covering providers, prompt construction, citation formatting/sanitization, and generator behavior across all 3 decision paths).
  - `tests/test_pipeline.py` (7 tests covering end-to-end pipeline execution on real corpus contradiction, gap, and supported cases, as well as CLI command execution).
- Antigravity assisted with updating `DECISIONS.md` (ADR-028 through ADR-033), `AI-USAGE.md`, and `README.md`.
- The project team verified that:
  - All 318 tests pass offline without an OpenAI API key.
  - GPT-4o mini is part of the runtime answer-generation architecture for `SUPPORTED` decisions only, and is never used for evidence evaluation or refusal overrides.
  - Citations are validated deterministically against authoritative retrieved clauses.

---

## Milestone 6 — Validation and Targeted Corrections

- Manual end-to-end testing against the real policy corpus exposed two issues:
  1. **Duplicate numeric rendering in conflict reports:** §4.3.2 repeated "10 calendar days" twice in its clause text, which was concatenated into `"10 calendar days, 10 calendar days"`. Antigravity resolved this by deduplicating raw numeric strings in `src/evidence/contradiction.py`.
  2. **Apparent-gap false SUPPORTED decision:** For "What is the policy for full-time students?", hybrid retrieval fetched §7.1.3 (which delegates to §5.4) alongside §5.4.1/§5.4.2 (care allowances). Section-level prefix matching in `src/evidence/evaluator.py` previously considered §5.4 resolved. Antigravity implemented `_is_ref_topically_resolved` to enforce that prefix-matched sub-clauses must share meaningful query vocabulary to count as resolving a delegation gap.
- Antigravity added regression test suites `TestApparentGapRegressions` and `TestConflictRenderingRegressions` in `tests/test_pipeline.py`.
- The project team verified that all 329 tests pass across the entire suite and that the student question deterministically yields `INSUFFICIENT` without LLM overrides.

