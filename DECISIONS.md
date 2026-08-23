# Architecture Decision Records

This file documents meaningful engineering decisions made during development.
Each entry records the decision, its context, alternatives considered, the
chosen rationale, and known trade-offs.

---

## ADR-001 — Language: Python 3.11

**Decision:** Use Python 3.11 as the implementation language.

**Context:** The problem requires combining document parsing, embedding models,
vector search, a language model API, and structured validation. The hackathon
stack (Sentence Transformers, FAISS, rank-bm25, OpenAI SDK, Pydantic, Typer)
is all first-class Python. Python 3.11 is the current stable release with
meaningful performance improvements over 3.10.

**Alternatives considered:**
- TypeScript/Node.js: strong LLM ecosystem but weaker support for FAISS and
  Sentence Transformers natively.

**Why Python:** Ecosystem fit is near-perfect. Every selected library has an
actively maintained Python package. The team has Python experience.

**Trade-offs:** Python is slower than compiled languages; this is not a
constraint for a CLI-based hackathon submission where latency is explicitly
not a requirement.

---

## ADR-002 — Modular package structure

**Decision:** Organise the source into distinct packages:
`ingestion`, `retrieval`, `evidence`, `generation`, `citation`, `models`.

**Context:** The core architectural requirement is that retrieval, evidence
evaluation, and answer generation are *separate stages*, not a single
pass-through pipeline. Modular packages enforce this boundary in code, make
each component independently testable, and allow judges to understand the
architecture by reading the directory structure alone.

**Alternatives considered:**
- Single flat module: easier initially but makes the three-stage separation
  invisible and harder to test in isolation.
- LangChain chains: would obscure the deliberate separation that the challenge
  is judging.

**Why modular packages:** Matches the architectural requirement exactly. Also
makes it straightforward to swap individual components (e.g., replace FAISS
with a different vector store) without touching unrelated code.

**Trade-offs:** Slightly more boilerplate at project initialisation. Worth it
for clarity.

---

## ADR-003 — CLI interface (Typer), not a web frontend

**Decision:** Use a Typer-based CLI as the user interface.

**Context:** The hackathon brief explicitly states that a web interface is not
required and that a CLI is sufficient. The central judging criterion is the
quality of grounding, refusal, and citation — not UI polish.

**Alternatives considered:**
- FastAPI REST service: would allow web clients but adds significant
  complexity and is outside the stated scope.
- Gradio / Streamlit: quick to build but adds a large dependency for no
  evaluation benefit.

**Why Typer CLI:** Minimal, readable, and standard for Python tools. Typer
produces clean `--help` output automatically. Keeps the focus on the RAG
pipeline rather than UI scaffolding.

**Trade-offs:** No browser-accessible interface. Acceptable given the brief.

---

## ADR-004 — Lightweight dependency set (no LangChain)

**Decision:** Use specific, purpose-built libraries instead of a high-level
framework such as LangChain or LlamaIndex.

**Context:** The challenge requires a transparent, explainable architecture.
High-level frameworks abstract away the retrieval → evidence evaluation →
answer generation separation that is the core of this submission. Judges
need to be able to follow the data flow.

**Alternatives considered:**
- LangChain: extensive ecosystem but opinionated chains obscure the deliberate
  three-stage decision logic.
- LlamaIndex: good for RAG but similarly opinionated; introduces many transitive
  dependencies.

**Why lightweight stack:** Each library has a single, clear responsibility.
The pipeline is readable without framework-specific knowledge. Dependencies
can be added later if a concrete requirement justifies them (documented in a
future ADR).

**Trade-offs:** More application-level glue code to write. Acceptable because
the glue code *is* the submission — the evidence evaluation logic is what is
being judged.

---

## ADR-005 — Markdown ingestion instead of PDF extraction

**Decision:** Read the policy corpus directly as UTF-8 Markdown text using
Python's standard `pathlib` / file I/O, with a custom structure-aware
Markdown parser. Do not use PyMuPDF or any PDF extraction library.

**Context:** The Brite Spark organizers supplied the policy manual as a
Markdown (`.md`) file, not a PDF. This was clarified after the initial
project setup. PyMuPDF was included in `requirements.txt` during Milestone 1
based on the original assumption that the corpus would be a PDF.

**Alternatives considered:**
- PyMuPDF: appropriate for PDF extraction but irrelevant when the source is
  already plain Markdown. Adding it would be unnecessary weight.
- `mistune` / `markdown-it-py`: full Markdown-to-HTML parsers. Useful for
  rendering but not needed here — we want to preserve headings and clause
  structure as text, not render to HTML.
- `python-markdown`: same concern as above.

**Why direct Markdown parsing:** The Markdown source already contains the
structure we need (headings, clause identifiers, numbered lists). A custom
parser that walks the heading hierarchy preserves clause boundaries faithfully
without any lossy format conversion. It is also simpler, has no additional
dependencies, and keeps the source text byte-for-byte identical to the
original corpus.

**Consequences:**
- PyMuPDF removed from `requirements.txt`.
- The `src/ingestion/` package will implement a Markdown-aware clause
  extractor rather than a PDF text extractor.
- The policy manual must be stored as a `.md` file in `src/data/raw/`.
- The source corpus must never be modified.
