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

---

## ADR-006 — Preserve raw Markdown exactly as loaded

**Decision:** The `PolicyDocument.raw_text` field stores the byte-for-byte
UTF-8 content of the source file.  The ingestion layer performs no
normalisation, whitespace stripping, heading rewriting, or any other
transformation.

**Context:** Later pipeline stages — retrieval, evidence evaluation, and
citation — must be able to trace every claim back to the exact source text.
If the ingestion layer modifies the text, traceability breaks and it becomes
impossible to show a reviewer the verbatim policy wording that supports an
answer.  The policy manual is also the authoritative source of truth for
contradictions; modifying it would obscure the planted inconsistency.

**Alternatives considered:**
- Strip leading/trailing whitespace per line: convenient for downstream regex
  matching, but loses positional fidelity and changes the character count.
- Normalise `—` (em dash) to `-`: loses document character fidelity.

**Why preserve exactly:** No transformation is needed at load time.  Any
normalisation required for pattern matching can be applied transiently within
the parser, without altering the stored raw text.

**Trade-offs:** Downstream code must handle original Markdown formatting
(bold markers, `§` prefix, etc.) rather than cleaned text.  This is
intentional and keeps responsibility clearly in the parser layer.

---

## ADR-007 — Separate ingestion from structural inspection

**Decision:** Implement `load_policy_document` (loader) and
`inspect_markdown` (inspector) as two separate functions in two separate
modules.

**Context:** Loading is a pure I/O concern: open a file, validate it, return
the content.  Structural inspection is a parsing concern: find headings,
count list items, detect clause-like patterns.  Mixing them would make both
harder to test in isolation and would couple file I/O changes to parsing
logic changes.

**Alternatives considered:**
- Single function that loads and inspects in one pass: simpler to call but
  harder to test, and couples two different responsibilities.

**Why separate:** Single-responsibility principle.  The loader is fully
testable without any Markdown structure knowledge.  The inspector can be
called on any `PolicyDocument`, including synthetic test documents that are
never on disk.  This separation also makes it straightforward to add
alternative inspection passes later without touching the loader.

**Trade-offs:** Callers need two calls instead of one.  Acceptable because
the CLI entry point will wrap both calls, and each component is individually
unit-testable.

---

## ADR-008 — Structural inspection before clause parser design

**Decision:** Run a structural inspection pass on the actual corpus
(Milestone 2) before designing the clause parser (Milestone 3).

**Context:** The policy manual structure was not known in advance.
Designing a parser from assumptions about "typical" Markdown policy documents
would risk building the wrong abstraction, especially given the requirement
to handle the planted inconsistency and gap correctly.

**Key observations from the actual corpus (Calder County HSP):**
- Document: 608 lines, 29 157 characters.
- Heading hierarchy: H1 = Part headings (12 Parts + 3 title lines);
  H2 = section headings (numbered N.N, e.g. "4.3 Recipient obligations").
  No H3 or deeper headings are used.
- Clause paragraphs: numbered **N.N.N** in bold at the start of each
  paragraph (137 clauses total).  These are the authoritative clause IDs.
- Sub-items within a clause: lettered **(a)**, **(b)**, … style (63 items).
  These are parts of a clause, not separate clauses.
- Tables: 14 table rows (income thresholds in §6.6, needs figures in §7.2).
- Cross-references: 53 unique `§N.N.N`-style references throughout.
- No unordered (bullet) list items.
- No code blocks.
- **Intentional contradiction identified:** §4.3.2 specifies **10 calendar
  days** as the reporting window for changes of circumstance.  §9.1.4
  cross-references §4.3 but states **30 calendar days** for the same
  obligation.  These cannot both be correct.  This must not be "fixed."
- **Apparent gap to investigate:** §7.1.3 refers to "full-time students
  (see §5.4)" but §5.4 addresses care allowances, not students.  The
  cross-reference appears to be incorrect; there is no dedicated student
  section in Part 5.  This is a candidate gap/inconsistency case.

**Why inspect first:** The observations above directly shape the Milestone 3
clause parser: clauses are delimited by bold **N.N.N** paragraph openers,
not by headings.  This would not have been clear from assumptions alone.

**Trade-offs:** One milestone is spent on inspection with no clause objects
produced.  This is the correct trade-off because a wrong parser design would
affect every downstream milestone.
