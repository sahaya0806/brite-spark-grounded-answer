# The Grounded Answer

**Hackathon:** Brite Spark 2026  
**Problem:** Problem 1 — The Grounded Answer  
**Category:** AI / RAG

## Description

The Grounded Answer is a CLI-based policy question-answering assistant. Given a plain-language question, it retrieves relevant clauses from a supplied policy manual, evaluates whether the evidence actually supports a definitive answer, and either produces a grounded, cited response or explicitly refuses when the policy does not settle the matter. Every substantive claim in an answer is traceable to a specific policy clause. The system deliberately separates retrieval, evidence sufficiency evaluation, and answer generation to avoid the dangerous failure mode of a fluent-but-unsupported response.

## Current Status

All 6 milestones are implemented and the 10-question final evaluation has been completed.

**Milestone 6 — Grounded Answer Generation** 
**Milestone 5 — Evidence Evaluation and Decision Layer** 
**Milestone 4 — Hybrid Policy Retrieval** 
**Milestone 3 — Clause-Level Parsing and Structured Clause Store** 
**Milestone 2 — Markdown Policy Ingestion** 
**Milestone 1 — Project Foundation** 

## Evaluation

The following 10 questions were run against the real supplied policy corpus.
No answers were hardcoded. Results are recorded honestly, including failures.

| # | Question | Expected | Actual | Result |
|---|----------|----------|--------|--------|
| 1 | What information must an applicant provide? | SUPPORTED | SUPPORTED | **PASS** |
| 2 | What evidence is required to establish an applicant's identity, residence, income, and resources? | SUPPORTED | SUPPORTED |  **PASS** |
| 3 | What are the recipient's obligations to report changes in circumstances? | Not specified | CONFLICTING |  **PASS** |
| 4 | What income threshold is used when assessing eligibility? | Not specified | INSUFFICIENT |  **PASS** |
| 5 | What income can be disregarded when calculating entitlement? | Not specified | SUPPORTED |  **FAIL** |
| 6 | How many days does a recipient have to report a change? | CONFLICTING | CONFLICTING |  **PASS** |
| 7 | What is the policy for full-time students? | INSUFFICIENT | INSUFFICIENT |  **PASS** |
| 8 | What is the policy for a household that owns three electric vehicles? | INSUFFICIENT | INSUFFICIENT |  **PASS** |
| 9 | Does the program provide a special benefit for households affected by flooding? | INSUFFICIENT | INSUFFICIENT |  **PASS** |
| 10 | What rule applies to full-time students under the policy? | INSUFFICIENT | SUPPORTED | **FAIL** |

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

# Ask a question (requires OPENAI_API_KEY)
python -m src ask "What is the resource limit for a household?"

# Test contradiction handling
python -m src ask "How many days does a recipient have to report a change?"

# Test gap/refusal handling
python -m src ask "What is the policy for full-time students?"
```

## Running the Test Suite

The test suite runs **100% offline** without requiring an OpenAI API key or network access:

```bash
pytest
```

Expected output: **all 318 tests pass** (in under 3 seconds).

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
