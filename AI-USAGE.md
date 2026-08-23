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
