# AI Usage Record

This file honestly documents the use of AI assistance during development of
The Grounded Answer. It is updated incrementally as the project progresses.

---

## Tools Used

### Kiro (coding assistant)
- Used for: code generation, file scaffolding, test writing, refactoring.
- All generated code is reviewed by the project team before committing.
- The team is responsible for verifying correctness, running tests, and
  making final implementation decisions.

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
