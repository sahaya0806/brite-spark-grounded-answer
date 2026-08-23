# The Grounded Answer

**Hackathon:** Brite Spark 2026  
**Problem:** Problem 1 — The Grounded Answer  
**Category:** AI / RAG

## Description

The Grounded Answer is a CLI-based policy question-answering assistant. Given a plain-language question, it retrieves relevant clauses from a supplied policy manual, evaluates whether the evidence actually supports a definitive answer, and either produces a grounded, cited response or explicitly refuses when the policy does not settle the matter. Every substantive claim in an answer is traceable to a specific policy clause. The system deliberately separates retrieval, evidence sufficiency evaluation, and answer generation to avoid the dangerous failure mode of a fluent-but-unsupported response.

## Current Status

**Milestone 6 — Grounded Answer Generation** ✅  
**Milestone 5 — Evidence Evaluation and Decision Layer** ✅  
**Milestone 4 — Hybrid Policy Retrieval** ✅  
**Milestone 3 — Clause-Level Parsing and Structured Clause Store** ✅  
**Milestone 2 — Markdown Policy Ingestion** ✅  
**Milestone 1 — Project Foundation** ✅

The end-to-end pipeline is fully implemented:
1. **Ingestion:** 137 authoritative clauses extracted with source line tracking.
2. **Retrieval:** Hybrid vector (FAISS + OpenAI `text-embedding-3-small`) and lexical (BM25Okapi) retrieval with Reciprocal Rank Fusion.
3. **Evidence Evaluation:** 3-way deterministic decision (`SUPPORTED`, `INSUFFICIENT`, `CONFLICTING`) with numeric/obligation conflict detection and gap analysis.
4. **Answer Generation:** Grounded plain-language synthesis via GPT-4o mini with strict clause citations for supported questions, and deterministic refusal/conflict reporting for non-supported questions.

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
