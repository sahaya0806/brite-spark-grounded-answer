# The Grounded Answer

**Hackathon:** Brite Spark 2026  
**Problem:** Problem 1 — The Grounded Answer  
**Category:** AI / RAG

## Description

The Grounded Answer is a CLI-based policy question-answering assistant. Given a plain-language question, it retrieves relevant clauses from a supplied policy manual, evaluates whether the evidence actually supports a definitive answer, and either produces a grounded, cited response or explicitly refuses when the policy does not settle the matter. Every substantive claim in an answer is traceable to a specific policy clause. The system deliberately separates retrieval, evidence sufficiency evaluation, and answer generation to avoid the dangerous failure mode of a fluent-but-unsupported response.

## Repository Branches

- **`main` (Git tag: `ogproj`):** Baseline Day-1 implementation. Evaluates queries strictly against the original 2025 Consolidated Policy Manual.
- **`surprise-challenge` (Current branch):** Day-2 implementation. Introduces deterministic temporal policy grounding, Amendment No. 2026-01 parsing, and CLI `--date` support.

To switch branches:
```bash
# Switch to the Day-2 Surprise Challenge branch (current)
git checkout surprise-challenge

# Switch to the original Day-1 baseline branch
git checkout main
```

---

## Current Status

All Day-1 milestones and Day-2 surprise challenge milestones are implemented and verified.

- **Day-1 Milestones 1–6:** Core Grounded QA Pipeline [COMPLETE]
- **Day-2 Milestone 1:** Temporal Policy Data Model [COMPLETE]
- **Day-2 Milestone 2:** Amendment Parsing & Structured Policy Versions [COMPLETE]
- **Day-2 Milestone 3:** Deterministic Temporal Applicability Layer [COMPLETE]
- **Day-2 Milestone 4:** Temporal Filter, Pipeline & CLI Integration [COMPLETE]

---

## Day-2 Surprise Challenge: Temporal Policy Grounding

### 1. Challenge Overview & Problem Definition

In Day 2 of Brite Spark 2026, the organizers introduced **Amendment No. 2026-01** (effective **1 March 2026**).

The core requirement is:
> **The assistant must answer policy questions correctly for the DATE OF THE CLAIM / DETERMINATION being asked about, rather than simply answering according to the currently effective policy or confusing different temporal versions.**

Under this requirement:
- A claim or change occurring **before 1 March 2026** must be evaluated under the original **2025 Consolidated Policy Manual**.
- A claim or determination occurring **on or after 1 March 2026** must be evaluated under the amended rules established by **Amendment No. 2026-01**.
- If a date-sensitive question is asked **without a date**, the system must **refuse safely** (`INSUFFICIENT`) and prompt for `--date YYYY-MM-DD` rather than guessing "today" or assuming March 1.
- Non-date-sensitive questions (unamended policy) must continue to function normally without requiring a date.

---

### 2. Where the Files Are Located

| Component | File Path | Purpose / Implementation |
|---|---|---|
| **Original Policy Manual** | `data/raw/policy_manual.md` | 608 lines, 137 clauses. **Never modified.** |
| **Amendment Document** | `data/raw/Amendment No. 2026-01.md` | 51 lines. **Never modified.** |
| **Temporal Data Model** | `src/ingestion/parser.py` | `PolicyClause` extended with `effective_from`, `effective_to`, `source_document`. |
| **Amendment Parser** | `src/ingestion/amendment.py` | `parse_amendment()`, `AmendmentDocument`, `AmendmentChange`, `TableRow`, `TriggerType`, `ChangeType`. |
| **Temporal Models** | `src/temporal/models.py` | `TemporalContext`, `ResolutionStatus`, `TemporalResolution`. |
| **Applicability Resolver** | `src/temporal/resolver.py` | `TemporalApplicabilityResolver` — deterministic rule engine evaluating §5.1 / §5.2 triggers. |
| **Temporal Filter Layer** | `src/temporal/filter.py` | `TemporalFilter` — maps raw retrieval results to applicable policy versions. |
| **Pipeline Integration** | `src/pipeline.py` | `PolicyQAPipeline` — unifies Retrieval → Temporal Filter → Evidence Evaluator → Generator. |
| **CLI Application** | `src/app.py` | Typer CLI with `--date / -d` and `--amendment` parameters. |
| **Citation Formatting** | `src/citation/renderer.py` | Alphanumeric IDs (§10.5.3A) and provenance tags (`(Amendment No. 2026-01)`). |
| **Unit & Integration Tests** | `tests/` | 409 automated offline tests (including `test_amendment_parser.py`, `test_temporal_resolver.py`, `test_temporal_pipeline.py`). |

---

### 3. Architecture of the Temporal Grounded QA System

```
                      User Question + [--date YYYY-MM-DD]
                                     ↓
      ┌─────────────────────────────────────────────────────────────┐
      │ 1. Hybrid Retrieval (FAISS Semantic + BM25 Lexical via RRF)  │
      │    Indexes 138 total clauses (137 base + §10.5.3A)          │
      └──────────────────────────────┬──────────────────────────────┘
                                     │ Candidate Clauses
                                     ↓
      ┌─────────────────────────────────────────────────────────────┐
      │ 2. Temporal Filter (TemporalFilter & Applicability Resolver)│
      │    • Determination Date Trigger (§5.1): §6.4.1, §6.6.1,     │
      │      §10.5.2, §10.5.3A                                      │
      │    • Change of Circumstances Trigger (§5.2): §4.3.2, §9.1.4 │
      │    • Replaces clause with applicable version; drops future; │
      │    • If date-sensitive clause lacks required date → REFUSE  │
      └──────────────────────────────┬──────────────────────────────┘
                                     │ Active Temporal Evidence
                                     ↓
      ┌─────────────────────────────────────────────────────────────┐
      │ 3. Evidence Evaluation (EvidenceEvaluator)                  │
      │    • Signal extraction & relevance scoring                  │
      │    • Scope-aware contradiction detection (on active policy) │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ↓                         ↓                         ↓
     [SUPPORTED]              [INSUFFICIENT]             [CONFLICTING]
           ↓                         ↓                         ↓
  Grounded Answer          Deterministic Refusal    Surfaces Historical
  (GPT-4o mini)            + Date Requirement       Contradictions
  + Exact Citations        Explanation              with Exact Lines
```

#### Key Architectural Decisions:
1. **Separation of Retrieval and Applicability:**
   Retrieval indexes all candidate provisions across both documents (138 clauses). The `TemporalFilter` resolves each candidate clause to its applicable version *before* evidence evaluation begins.
2. **Elimination of Cross-Temporal False Conflicts:**
   The pre-amendment 10-day rule (§4.3.2) and post-amendment 14-day rule (§4.3.2) are never evaluated simultaneously. For any given claim date, only the temporally active rule is presented to the contradiction detector.
3. **The Two Date Triggers (§5.1 vs §5.2):**
   - **Determination Date Trigger (§5.1):** Governs earnings disregards (§6.4.1), income thresholds (§6.6.1), and sanctions (§10.5.2, §10.5.3A).
   - **Change of Circumstances Trigger (§5.2):** Governs change reporting deadlines (§4.3.2) and overpayment safe harbours (§9.1.4).
4. **Strict Refusal vs. Guessing Boundary:**
   If a date-sensitive clause is retrieved but the caller provided no `--date`, the pipeline refuses with status `[INSUFFICIENT]` and clearly explains that a date parameter is required. It **never** guesses "today" or defaults to March 1.

---

### 4. How to Run and Test the Day-2 System

#### Run All 409 Automated Offline Tests
```bash
pytest
```
*Runs 100% offline in under 8 seconds without network calls or API keys.*

#### Test 1: Change Reporting Deadline Pre-Amendment (`2026-02-20`)
*Evaluates against 2025 manual where §4.3.2 (10 days) and §9.1.4 (30 days) conflicted:*
```bash
python -m src ask "How many days does a recipient have to report a change?" --date 2026-02-20
```
- **Result:** `Status: [CONFLICTING]`
- **Answer:** Surfaces the contradiction between §4.3.2 (10 calendar days) and §9.1.4 (30 calendar days).

#### Test 2: Change Reporting Deadline Post-Amendment (`2026-04-20`)
*Evaluates against Amendment No. 2026-01 where both provisions are aligned to 14 days:*
```bash
python -m src ask "How many days does a recipient have to report a change?" --date 2026-04-20
```
- **Result:** `Status: [SUPPORTED]`
- **Answer:** 14 calendar days.
- **Citation:** `§4.3.2, line 18 (Amendment No. 2026-01)`

#### Test 3: Earnings Disregard Pre-Amendment (`2026-02-20`)
```bash
python -m src ask "What is the earnings disregard under section 6.4.1?" --date 2026-02-20
```
- **Result:** `Status: [SUPPORTED]`
- **Answer:** $120 per month.
- **Citation:** `§6.4.1, line 280 (policy_manual.md)`

#### Test 4: Earnings Disregard Post-Amendment (`2026-04-20`)
```bash
python -m src ask "What is the earnings disregard under section 6.4.1?" --date 2026-04-20
```
- **Result:** `Status: [SUPPORTED]`
- **Answer:** $175 per month.
- **Citation:** `§6.4.1, line 23 (Amendment No. 2026-01)`

#### Test 5: Date-Sensitive Question Without Date (Safe Refusal)
```bash
python -m src ask "How many days does a recipient have to report a change?"
```
- **Result:** `Status: [INSUFFICIENT]`
- **Answer:** Refuses safely, explicitly prompting the user to supply `--date YYYY-MM-DD`.

#### Test 6: Unamended Question Without Date (Day-1 Compatibility)
```bash
python -m src ask "What is the resource limit for a household?"
```
- **Result:** `Status: [SUPPORTED]`
- **Answer:** $4,000 (answers normally without needing a date parameter).

---

## Evaluation

The following 10 questions were run against the real supplied policy corpus.
No answers were hardcoded. Results are recorded honestly, including failures.

| # | Question | Expected | Actual | Result |
|---|----------|----------|--------|--------|
| 1 | What information must an applicant provide? | SUPPORTED | SUPPORTED | PASS |
| 2 | What evidence is required to establish an applicant's identity, residence, income, and resources? | SUPPORTED | SUPPORTED | PASS |
| 3 | What are the recipient's obligations to report changes in circumstances? | Not specified | CONFLICTING | PASS |
| 4 | What income threshold is used when assessing eligibility? | Not specified | INSUFFICIENT | PASS |
| 5 | What income can be disregarded when calculating entitlement? | Not specified | SUPPORTED | FAIL |
| 6 | How many days does a recipient have to report a change? | CONFLICTING | CONFLICTING | PASS |
| 7 | What is the policy for full-time students? | INSUFFICIENT | INSUFFICIENT | PASS |
| 8 | What is the policy for a household that owns three electric vehicles? | INSUFFICIENT | INSUFFICIENT | PASS |
| 9 | Does the program provide a special benefit for households affected by flooding? | INSUFFICIENT | INSUFFICIENT | PASS |
| 10 | What rule applies to full-time students under the policy? | INSUFFICIENT | SUPPORTED | FAIL |

**Total questions: 10 | Passed: 8 | Failed: 2 | Pass rate: 80%**

Test suite (333 automated offline tests): **333 passed, 0 failed**.

## What the System Does

```
Question
  → Hybrid Retrieval (FAISS semantic + BM25 lexical, merged via RRF)
  → Evidence Evaluation (deterministic signal extraction, 3-way decision)
  ├── SUPPORTED    → GPT-4o mini grounded answer + exact clause citations
  ├── INSUFFICIENT → Deterministic refusal + gap explanation
  └── CONFLICTING  → Deterministic conflict report (surfacing both provisions)
```

The system:
- Reads the supplied Markdown policy corpus directly without modification.
- Parses the corpus into 137 identifiable policy clauses with source line tracking.
- Retrieves at clause level (not arbitrary chunks) preserving citation accuracy.
- Uses FAISS for semantic retrieval and BM25 for lexical retrieval.
- Combines results using Reciprocal Rank Fusion (RRF).
- Evaluates evidence sufficiency deterministically — separate from retrieval.
- Supports SUPPORTED, INSUFFICIENT, and CONFLICTING outcomes.
- Detects the known contradiction between §4.3.2 (10 calendar days) and §9.1.4 (30 calendar days).
- Provides clause-level citations with exact source line numbers.
- Uses GPT-4o mini for natural-language answer construction on SUPPORTED cases only.
- Refuses instead of guessing when evidence is insufficient.
- Surfaces both conflicting provisions when the manual contradicts itself.

## Known Limitations

- **Paraphrase sensitivity:** Different phrasings of the same question can retrieve
  different clauses and produce different decisions (Q7 passes, Q10 fails for the
  same gap about full-time students).
- **Lexical signal precision:** The evidence evaluator uses vocabulary overlap
  signals. This can score income-related clauses as sufficient for an income
  disregard question when semantically they do not answer it (Q5 failure).
- **Corpus-specific design:** The system is calibrated for the supplied Calder
  County Household Support Program corpus and is not a general document QA system.
- **No multi-clause reasoning:** Questions requiring reasoning across two or more
  distant, non-adjacent clauses may not be handled correctly.
- **No web UI:** CLI only. Intended for demonstration and evaluation.

## Future Improvements

1. **Improve retrieval/evidence performance** — especially for paraphrased questions,
   apparent gaps, cross-reference reasoning, and difficult multi-clause questions.
2. **Expand evaluation coverage** — 30–50 questions covering all 12 policy parts,
   adversarial phrasings, and edge cases.
3. **Better citation navigation** — source-line highlighting for quick verification.
4. **Simple UI** — a minimal Gradio/Streamlit interface for non-technical stakeholders,
   only after retrieval and evidence quality improvements are complete.



## Local Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (for live production retrieval & generation; test suite runs 100% offline)

### 1. Clone the repository

```bash
git clone https://github.com/sahaya0806/brite-spark-grounded-answer.git
cd brite-spark-grounded-answer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

> `.env` is git-ignored and must never be committed.

## Policy Corpus

The policy manual is the **Calder County Household Support Program** policy document,
supplied by the Brite Spark organisers as a Markdown file.

- Location: `data/raw/policy_manual.md`
- Format: UTF-8 Markdown
- The source file is never modified by the application.
- 608 lines, ~29 000 characters, 12 Parts, 137 numbered clauses.

## Running the Application

```bash
# Show system status
python -m src info

# Ask a standard question (requires OPENAI_API_KEY)
python -m src ask "What is the resource limit for a household?"

# Ask a date-aware question (Day 2 — Amendment No. 2026-01):
# Pre-amendment date (evaluates against 2025 consolidated manual):
python -m src ask "How many days does a recipient have to report a change?" --date 2026-02-20

# Post-amendment date (evaluates against Amendment No. 2026-01):
python -m src ask "How many days does a recipient have to report a change?" --date 2026-04-20

# Pre-amendment earnings disregard ($120/month):
python -m src ask "What is the earnings disregard?" --date 2026-02-20

# Post-amendment earnings disregard ($175/month):
python -m src ask "What is the earnings disregard?" --date 2026-04-20

# Date-sensitive question without --date (safely refuses and prompts for date):
python -m src ask "How many days does a recipient have to report a change?"
```

## Running the Test Suite

The test suite runs **100% offline** without requiring an OpenAI API key or network access:

```bash
pytest
```

Expected output: **all 409 tests pass** (in under 8 seconds).

## End-to-End Pipeline API

```python
from src.pipeline import PolicyQAPipeline
from src.generation.providers import OpenAIChatProvider
from src.retrieval.embeddings import OpenAIEmbeddingProvider

# Build end-to-end pipeline from Markdown policy corpus
pipeline = PolicyQAPipeline.build_from_corpus(
    corpus_path="data/raw/policy_manual.md",
    embedding_provider=OpenAIEmbeddingProvider(),
    chat_provider=OpenAIChatProvider(),
)

# Ask a question
answer = pipeline.ask("How many days does a recipient have to report a change?")

print(f"Status: {answer.status.value}")      # CONFLICTING
print(f"Answer: {answer.answer_text}")
print(f"Citations: {answer.citations}")
```

## Configuration

| Env Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for live OpenAI embeddings and chat generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat completion model |

| Component | Implementation |
|---|---|
| Ingestion | Structure-aware Markdown parsing into 137 `PolicyClause` records |
| Semantic Retrieval | FAISS `IndexFlatIP` + OpenAI `text-embedding-3-small` |
| Lexical Retrieval | BM25Okapi (`rank-bm25`) |
| Hybrid Merging | Reciprocal Rank Fusion ($k=60$) |
| Evidence Evaluation | Deterministic signal extraction & scope-aware conflict detection |
| Answer Generation | Grounded GPT-4o mini synthesis + deterministic refusal/conflict paths |
| Citation Validation | Deterministic extraction and validation against authoritative clause IDs |

## Project Structure

```text
src/
  app.py            # CLI entry point (Typer)
  pipeline.py       # PolicyQAPipeline — unified end-to-end RAG pipeline
  ingestion/
    loader.py       # load_policy_document() → PolicyDocument
    inspector.py    # inspect_markdown() → MarkdownInspection
    parser.py       # parse_clauses() → list[PolicyClause]
    store.py        # ClauseStore — in-memory clause index
  retrieval/
    embeddings.py   # EmbeddingProvider protocol + OpenAI / Fake implementations
    vector.py       # VectorIndex (FAISS semantic search)
    lexical.py      # LexicalIndex (BM25 keyword search)
    hybrid.py       # HybridRetriever (RRF merging) + RetrieverConfig
    models.py       # RetrievalResult
  evidence/
    models.py       # DecisionStatus, EvidenceItem, ConflictDetail, EvidenceDecision
    scoring.py      # Signal extraction, numeric fact extraction, relevance scoring
    contradiction.py# detect_conflicts() — topic & numeric conflict detector
    evaluator.py    # EvidenceEvaluator — 3-way decision logic
  generation/
    models.py       # GroundedAnswer data model
    providers.py    # ChatProvider protocol + OpenAI / Fake implementations
    prompts.py      # System and user prompts for strict grounded synthesis
    generator.py    # GroundedAnswerGenerator (SUPPORTED / INSUFFICIENT / CONFLICTING)
  citation/
    renderer.py     # Clause citation formatting, extraction, sanitization, validation
data/
  raw/              # Source policy document (policy_manual.md)
  processed/        # Parsed clause store
tests/              # Complete pytest test suite (318 offline tests)
```

## Architecture Overview

```
Question
   ↓
Hybrid Retrieval (Semantic + BM25)
   ↓
Candidate Clauses (RetrievalResult)
   ↓
Evidence Evaluation (EvidenceEvaluator)
   ├── SUPPORTED    → Grounded Answer Generator (GPT-4o mini) + Exact Citations
   ├── INSUFFICIENT → Deterministic Refusal + Gap Explanation + Escalation Guidance
   └── CONFLICTING  → Deterministic Conflict Report (surfacing both provisions)
```
