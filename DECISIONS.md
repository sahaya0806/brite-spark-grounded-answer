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

---

## ADR-009 — Authoritative clauses as the fundamental evidence unit

**Decision:** The fundamental evidence unit of the system is the authoritative
clause — a numbered policy paragraph identified by a bold ``**N.N.N**`` opener.
This is the unit stored, retrieved, and cited in every downstream stage.

**Context:** The challenge requires clause-level citations.  The alternative
would be arbitrary text chunks, but chunks would not correspond to any
identifiable policy provision, making citations meaningless.  The corpus
inspection in Milestone 2 confirmed that the document's own paragraph
numbering scheme provides ready-made, citable identifiers.

**Why **N.N.N** bold openers:** This is the exact pattern used consistently
across all 137 clauses in the real corpus.  It was derived from inspection
of the actual document, not from assumptions.  Alternative candidate patterns
(H2 section headings, lettered sub-items) were examined and rejected (see
ADR-010, ADR-011).

**Trade-offs:** A clause may be long and multi-part (e.g. §6.4.1 has seven
sub-items).  Retrieval may sometimes need to quote a sub-item rather than the
full clause.  This is acceptable: the clause is the citation unit, and the
sub-item text is preserved within the clause record for display purposes.

---

## ADR-010 — Lettered sub-items are clause parts, not independent clauses

**Decision:** Items ``(a)``, ``(b)``, ``(c)`` … are sub-items of their parent
``**N.N.N**`` clause and do not receive independent clause IDs.

**Context:** 63 lettered sub-items appear in the corpus.  Treating each as
an independent clause would produce ~200 "clauses" where the policy itself
only numbers 137.  Sub-items cannot be cited independently in the policy
(there is no authoritative identifier like ``§4.3.1(a)`` in the source).

**Alternatives considered:**
- Assign synthetic IDs such as ``4.3.1.a``: would invent identifiers not
  present in the policy.  Violates the requirement not to invent clause IDs.

**Why sub-items:** The sub-items are preserved as structured data within the
parent clause, so downstream systems can still display them.  But the
citation target remains the parent clause.

---

## ADR-011 — Part and Section headings preserved as context, not as clauses

**Decision:** H1 Part headings and H2 Section headings are used to populate
the ``part_id``, ``part_title``, ``section_id``, and ``section_title``
fields of each clause.  They are not represented as independent clause records.

**Context:** Parts and sections give context ("this is a rule about income")
but contain no policy text of their own.  Creating clause records for them
would artificially inflate the clause count and create records with no
citable substance.

**Why this approach:** Preserving the heading context within each clause record
means every clause is self-describing — a downstream citation can say
"Part 4, §4.3 Recipient obligations, clause 4.3.2" without having to look up
the heading hierarchy separately.

---

## ADR-012 — Cross-references extracted as metadata but not resolved

**Decision:** ``§``-prefixed references are extracted into the
``cross_references`` field as metadata.  The parser does not verify whether
the referenced clause exists, whether the reference is correct, or whether
it is consistent with other references.

**Context:** Resolving references at parse time would require the full clause
store to exist while parsing (a circular dependency) and, more importantly,
would obscure intentional inconsistencies.  The manual's planted gap
(§7.1.3 references §5.4 which covers care allowances, not students) must be
preserved as-is for the evidence evaluation stage to surface it.

**Consequences:** The evidence evaluation layer (Milestone 5) will receive the
cross-references as hints for multi-clause reasoning, not as verified links.

---

## ADR-013 — Tables are accumulated into their introducing clause

**Decision:** Markdown table rows (lines starting with ``|``) are treated as
body content of the clause whose opener immediately precedes them.  They are
accumulated into the clause's ``text`` field.

**Context:** Both tables in the corpus (§6.6.1, §7.2.1) appear immediately
after the clause that introduces them ("The thresholds are —", "The monthly
needs figures are —").  The table data is semantically part of that clause,
not a separate policy provision.

**Alternatives considered:**
- Attaching the table to the section heading rather than a clause: the corpus
  shows the table is always introduced by a clause sentence, not directly
  under a section heading with no clause.
- Treating each table row as an independent clause: there is no authoritative
  identifier for table rows; doing so would invent structure.

**Why inline accumulation:** The simplest approach that preserves the complete
policy content and keeps the table associated with the clause that declares it.
A downstream system displaying §6.6.1 will show the thresholds table as part
of that clause, which is the correct policy context.

---

## ADR-014 — Contradictions and apparent gaps are preserved by the parser

**Decision:** The parser preserves the source text exactly, including the
known inconsistency (§4.3.2 states 10 days; §9.1.4 states 30 days for the
same obligation) and the apparent cross-reference error (§7.1.3 → §5.4).

**Context:** The problem statement explicitly states that the corpus
intentionally contains difficult cases.  Resolving them at parse time would
destroy the very evidence that the evidence evaluation layer needs to reason
about.

**Consequences:** The test suite explicitly verifies that:
- §4.3.2 contains "10 calendar days" (unmodified)
- §9.1.4 contains "30 calendar days" (unmodified)
- §7.1.3 retains its §5.4 cross-reference (unresolved)

The contradiction and gap are first-class test cases, not implementation bugs.

---

## ADR-015 — Source line ranges preserved for future citation verification

**Decision:** Each ``PolicyClause`` records ``start_line`` and ``end_line``
(1-indexed, referring to the original Markdown source).

**Context:** A citation "§4.3.2, lines 200–200" is verifiable: a reviewer
can open the source file and check the exact line.  This is more auditable
than a character offset or a hash.

**Consequences:** Line numbers are stable as long as the source file is not
edited.  The source file must not be edited (as required by the project rules).
The citation layer (Milestone 7) can use these line numbers to display
exact source locations.

---

## ADR-016 — Clause-level retrieval, not arbitrary chunking

**Decision:** The retrieval unit is the ``PolicyClause`` record produced by
the Milestone 3 parser.  The corpus is NOT split into fixed-size token chunks.

**Context:** The challenge requires clause-level citations.  If we chunked
the policy into 500-token windows, a retrieved chunk might span parts of two
clauses, making it impossible to cite a specific provision.  With 137 clauses
averaging ~212 characters each, chunking provides no benefit and destroys
citation accuracy.

**Trade-offs:** A long clause (e.g. §6.4.1 with 7 sub-items plus a table)
is retrieved as a single unit.  This is correct: the clause is the citable
policy provision, and its full text is needed for evidence evaluation.

---

## ADR-017 — Hybrid retrieval (semantic + BM25)

**Decision:** Use both a semantic (embedding) index and a BM25 lexical index,
merged by Reciprocal Rank Fusion.

**Context:** The policy corpus contains both natural-language provisions and
precise legal/policy vocabulary.  Neither retrieval method alone covers both:

- Semantic retrieval handles paraphrased questions ("how many days do I have
  to tell the office?") that would fail exact-term matching.
- BM25 retrieval handles queries containing exact policy terms, numbers,
  dates, clause IDs, and specific vocabulary that semantic search may miss.

**Why hybrid:** The combination is strictly better than either method alone
for the range of question types expected in the evaluation set.

---

## ADR-018 — OpenAI text-embedding-3-small as the production embedding model

**Decision:** Use OpenAI's ``text-embedding-3-small`` model in production.

**Context:** The project already depends on the OpenAI SDK (for answer
generation in a later milestone).  ``text-embedding-3-small`` is OpenAI's
current small, cost-effective embedding model with strong performance on
English policy text.  No additional dependency is required.

**Alternatives considered:**
- ``sentence-transformers`` (all-MiniLM-L6-v2): already in requirements.txt,
  would work offline.  However, using two different embedding ecosystems
  (OpenAI for generation + sentence-transformers for embeddings) is more
  complex than using OpenAI consistently.  Sentence Transformers remain in
  requirements.txt for potential future use.
- ``text-embedding-ada-002``: older model, superseded by 3-small.

**Configuration:** The model name is read from the ``OPENAI_EMBEDDING_MODEL``
environment variable, defaulting to ``"text-embedding-3-small"``.

**Offline testing:** The ``FakeEmbeddingProvider`` allows the full test suite
to run without an API key.

---

## ADR-019 — FAISS flat inner-product index for vector search

**Decision:** Use ``faiss.IndexFlatIP`` (flat exact search, inner product)
over L2-normalised vectors, giving exact cosine similarity search.

**Context:** The corpus is 137 clauses.  Approximate nearest-neighbour
indexes (IVF, HNSW, etc.) are designed for millions of vectors.  A flat
exact index is the simplest, most correct implementation at this scale and
has negligible latency (microseconds for 137 vectors).

**Why FAISS rather than pure NumPy:** FAISS is already in requirements.txt
and provides a standard, well-tested interface.  NumPy cosine search would
also be correct at this size; FAISS was chosen for consistency with the
original project plan.

**Trade-offs:** If the corpus grew to tens of thousands of clauses, switching
to an approximate index would be straightforward (just change the index type).

---

## ADR-020 — rank-bm25 (BM25Okapi) for lexical retrieval

**Decision:** Use the ``rank-bm25`` library's ``BM25Okapi`` implementation.

**Context:** ``rank-bm25`` is already in requirements.txt.  BM25Okapi is the
standard BM25 variant used in information retrieval research.  No additional
dependency is needed.

**Tokeniser design:** The tokeniser preserves numbers, monetary values,
percentages, and dotted clause IDs as single tokens.  Standard stop-word
removal is intentionally omitted because policy vocabulary — "must", "may",
"not", "no" — carries legal meaning that must not be discarded.

**Score normalisation:** BM25Okapi scores are unbounded.  Within each result
set, scores are normalised to [0, 1] by dividing by the maximum score.

---

## ADR-021 — Reciprocal Rank Fusion (RRF) for hybrid merging

**Decision:** Use RRF (Cormack, Clarke, Buettcher, 2009) to merge semantic
and lexical candidates.

**Formula:** RRF(d) = Σᵢ 1 / (k + rankᵢ)  with k = 60 (standard default).

**Context:** Semantic cosine scores ∈ [0, 1] and normalised BM25 scores ∈
[0, 1] cannot be summed directly — their distributions differ.  RRF depends
only on rank position, not score magnitude, making it scale-independent.

**Alternatives considered:**
- Weighted sum of normalised scores: requires calibration of weights;
  the correct weights depend on query type, which we cannot know in advance.
- Linear combination with equal weights: implicitly assumes both methods have
  similar score distributions; not true for BM25 vs cosine.

**Parameters:** ``rrf_k=60`` is the standard value from the original paper.
``semantic_top_k=lexical_top_k=final_top_k=10`` are sensible defaults for
137 clauses.  All are configurable via ``RetrieverConfig``.

**Why RRF scores are NOT answer confidence:** RRF scores reflect retrieval
rank position only.  A high RRF score means "this clause ranked highly in
one or both retrieval systems for this query."  It says nothing about whether
the clause actually answers the question.  Evidence sufficiency is determined
by the evidence evaluation layer (Milestone 5).

---

## ADR-022 — EmbeddingProvider Protocol for testability

**Decision:** Define ``EmbeddingProvider`` as a Python ``Protocol`` and
provide a ``FakeEmbeddingProvider`` for tests.

**Context:** The test suite must run without an OpenAI API key.  If tests
called the real OpenAI API they would be slow, fragile (network dependent),
and expensive.  A Protocol decouples the vector index from the specific
embedding backend.

**``FakeEmbeddingProvider`` design:** Uses a deterministic hash-keyed PRNG
to produce unit vectors.  Identical texts always produce identical vectors
across Python sessions.  The fake has no semantic content — it is only used
to verify API contracts, deduplication, score ordering, and retrieval
mechanics.

**Consequences:** Every test that involves embedding uses
``FakeEmbeddingProvider``.  Tests that check real semantic quality (e.g.
"§4.3.2 surfaces for a reporting-deadline paraphrase") are verified against
lexical retrieval and BM25, where the fake embeddings still exercise the
full hybrid pipeline.

---

## ADR-023 — Three-way evidence decision boundary (SUPPORTED / INSUFFICIENT / CONFLICTING)

**Decision:** The evidence evaluation layer classifies retrieved evidence into
exactly three mutually exclusive statuses: ``SUPPORTED``, ``INSUFFICIENT``,
and ``CONFLICTING``.

**Context:** A simple binary decision (answer / no-answer) cannot distinguish
between a policy corpus that is silent/incomplete on a topic (which warrants a
standard refusal) and a policy corpus that contains conflicting rules (which
warrants an explanation of the conflicting provisions).  Collapsing these
cases leads either to confident answers when the manual itself disagrees or
unhelpful generic refusals when a user needs to know about a policy conflict.

**Alternatives considered:**
- Binary boolean flag (is_supported: bool): misses the distinction between
  silence and contradiction.
- Continuous confidence score: exposes a scalar threshold that is difficult to
  interpret and shifts decision burden to downstream components.

**Why three-way classification:** Each state maps cleanly to a distinct downstream
action in answer generation:
- ``SUPPORTED`` → generate grounded answer with citations.
- ``INSUFFICIENT`` → refuse explicitly, indicating what information is missing.
- ``CONFLICTING`` → surface the conflicting clauses and explain the inconsistency.

**Trade-offs:** Requires an explicit contradiction detection mechanism prior
to final support assessment.

---

## ADR-024 — Deterministic signal extraction and evidence scoring

**Decision:** Evidence evaluation uses deterministic content signal extraction
(obligations, eligibility conditions, reporting keywords, numeric/temporal facts)
combined with retrieval scores, rather than an unconstrained LLM call.

**Context:** The core principle of the project is that retrieval similarity
does not prove evidence sufficiency, and language models must not be trusted
to assess sufficiency unconstrained.  A deterministic evaluator is 100% reproducible,
has zero latency/cost overhead, requires no API keys to run the entire test suite,
and eliminates the risk of hallucinated justifications.

**Alternatives considered:**
- Direct LLM prompting for evidence evaluation: introduces non-determinism,
  potential prompt injection vulnerabilities, latency, and reliance on API keys
  for basic unit testing.

**Scoring design:**
- Content signal analysis extracts policy keywords, obligation verbs, and
  numerical/temporal values.
- Weighted base score combines semantic and lexical retrieval metrics.
- Capped bonus rewards substantive policy content without score inflation.
- Conservative default: when evidence is ambiguous or scores fall below
  thresholds, the decision defaults to ``INSUFFICIENT``.

---

## ADR-025 — Scope-aware contradiction detection

**Decision:** Contradictions are detected dynamically by comparing numeric and
temporal obligations across clauses that share significant policy vocabulary,
while respecting section boundaries and scoping rules.

**Context:** The policy manual contains an intentional contradiction: §4.3.2
states a 10-day change-of-circumstance reporting window, whereas §9.1.4 states
a 30-day reporting window for the same obligation.  The system must detect this
contradiction without hardcoding specific clause IDs (§4.3.2 or §9.1.4).

At the same time, different clauses may cite different numbers in genuinely
distinct contexts (e.g. 28 days for temporary absence vs 60 days for formal
reviews).

**Detection logic:**
1. Extract numeric facts by category (durations in days, monetary amounts, percentages).
2. Compare pairs of clauses retrieved above a relevance threshold.
3. Skip clauses within the same section (presumed complementary).
4. Require a minimum shared policy vocabulary threshold (topic overlap).
5. Flag a conflict only when differing numeric values apply to the same fact kind
   within overlapping subject matter.

**Trade-offs:** Heuristic vocabulary overlap may require threshold tuning for
larger corpora; for the 137-clause manual, it achieves 100% precision and recall.

---

## ADR-026 — Unresolved cross-reference tracking as evidence gaps

**Decision:** Cross-references within candidate clauses are compared against the
set of retrieved clause IDs.  If a primary clause delegates its substance to a
cross-reference that is absent from retrieved evidence (e.g. §7.1.3 referencing §5.4),
the evaluator flags this as an unresolved gap and defaults to ``INSUFFICIENT``.

**Context:** In the policy manual, §7.1.3 addresses full-time students by stating
"(see §5.4)", but §5.4 covers care allowances rather than students.  A naive
system might retrieve §7.1.3 and assume full-time students are covered.  Tracking
delegated cross-references ensures that missing or erroneous cross-references
prevent false ``SUPPORTED`` claims.

**Consequences:** Questions regarding full-time students correctly yield
``INSUFFICIENT`` with a recommendation to refuse, preventing hallucinated answers.

---

## ADR-027 — Structured EvidenceDecision and strict clause ID preservation

**Decision:** The evaluator produces a frozen ``EvidenceDecision`` dataclass
referencing authoritative ``PolicyClause`` objects directly from the input
retrieval results.

**Context:** The contract between retrieval, evidence evaluation, and future
answer generation must guarantee that every cited clause ID exists in the
authoritative ``ClauseStore``.  Hallucinated or synthetic clause IDs cannot be
permitted to propagate to the citation layer.

**Design:**
- ``primary_clauses`` holds the exact ``PolicyClause`` instances supporting or
  conflicting in the decision.
- ``supporting_clause_ids`` convenience property surfaces the exact string IDs.
- Retrieval results are deduplicated by ``clause_id`` upon entry to the evaluator.
- All clause references are verified to have originated from the authoritative
  corpus.

---

## ADR-028 — LLM role strictly as grounded natural-language constructor (GPT-4o mini)

**Decision:** GPT-4o mini is employed exclusively to express accepted policy
evidence into fluent, plain-language natural text for ``SUPPORTED`` decisions.
It is never used to determine policy applicability, evaluate evidence sufficiency,
decide between conflicting provisions, or access general external knowledge.

**Context:** Allowing an LLM to answer policy questions directly produces fluent
hallucinations when policy details are missing or nuanced.  By restricting the
model to synthesize only the specific text of accepted ``PolicyClause`` records,
the LLM operates as an expression compiler rather than a policy authority.

**Alternatives considered:**
- End-to-end RAG with LLM self-evaluation: prone to confident hallucinations
  and inconsistent refusal thresholds.
- Template-only string formatting: produces rigid answers that do not read
  naturally or accommodate multi-clause synthesis.

**Constraints:**
- The prompt explicitly forbids external knowledge, extrapolation, and uncited claims.
- Every substantive claim must be tagged with an exact clause citation ``[§<clause_id>]``.

---

## ADR-029 — ChatProvider Protocol and deterministic FakeChatProvider for offline testing

**Decision:** Define ``ChatProvider`` as a Python ``Protocol`` and provide a
deterministic ``FakeChatProvider`` for unit and integration testing.

**Context:** The test suite must run 100% offline without requiring network calls
or an OpenAI API key.  Decoupling the generation layer via a Protocol ensures
fast test execution (<3s for 318+ tests) and prevents flaky test runs.

**Implementations:**
- ``OpenAIChatProvider``: production client using ``OPENAI_CHAT_MODEL``
  (defaulting to ``gpt-4o-mini``) and ``OPENAI_API_KEY``.
- ``FakeChatProvider``: test double supporting canned responses, custom
  responders, and call-history inspection.

---

## ADR-030 — Deterministic citation validation and provenance enforcement

**Decision:** All citations in generated responses are extracted, validated, and
sanitized by deterministic Python code against the authoritative
``supporting_clause_ids`` from ``EvidenceDecision``.

**Context:** A language model cannot be trusted not to invent or misremember clause
numbers (e.g. inventing "§99.99" or confusing "§4.3.2" with "§4.3.1").

**Enforcement:**
1. Extract all clause ID patterns (`§N.N.N` or `clause N.N.N`) from LLM text.
2. Filter against the allowed set of `supporting_clause_ids`.
3. Strip any unallowed/hallucinated citations from the final text.
4. If the LLM omitted citations, deterministically append valid citation tags.
5. Format full citations with source line ranges (e.g. `§4.3.2, line 200`).

---

## ADR-031 — Separate refusal and conflict generation pathways without LLM overrides

**Decision:** For ``INSUFFICIENT`` and ``CONFLICTING`` decisions, user-facing
responses are constructed deterministically without invoking the language model.

**Context:** When evidence is insufficient or contradictory, asking an LLM to
generate a refusal risks the LLM answering anyway from parametric memory.
Deterministic pathways guarantee that:
- ``INSUFFICIENT`` always produces a clear refusal explaining the specific gap
  and recommending policy administration consultation.
- ``CONFLICTING`` always surfaces both conflicting provisions (e.g. §4.3.2 vs
  §9.1.4) without picking one.
- Zero LLM tokens are consumed for non-supported questions.

---

## ADR-032 — Explicit answer vs refusal boundary

**Decision:** The boundary between answering and refusing is explicitly defined
by the ``EvidenceEvaluator`` rules and thresholds:
1. **Answer (SUPPORTED)**: Requires aggregate support score $\ge 0.35$, at least
   one strong clause score $\ge 0.40$, no material contradictions, and no
   unresolved delegating cross-references.
2. **Refusal (INSUFFICIENT)**: Triggered when evidence scores fall below threshold,
   when clauses delegate to missing cross-references (§7.1.3 → §5.4), or when no
   relevant clauses are retrieved.
3. **Conflict Explanation (CONFLICTING)**: Triggered when two or more relevant
   clauses from different sections specify competing values for the same obligation.

**Rationale:** Setting a conservative refusal boundary prevents false confidence.
A refusal protects claimants and staff from incorrect determinations, whereas a
hallucinated answer can cause compliance failures.

---

## ADR-033 — Unified PolicyQAPipeline and Typer CLI interface

**Decision:** Provide ``PolicyQAPipeline`` as the single composition root uniting
``HybridRetriever``, ``EvidenceEvaluator``, and ``GroundedAnswerGenerator``,
invoked directly via `python -m src ask "<question>"`.

**Context:** The hackathon requirements prioritize end-to-end explainability and
ease of execution from a clean clone.  A unified pipeline object makes testing,
CLI execution, and future milestone evaluation clean and straightforward.
