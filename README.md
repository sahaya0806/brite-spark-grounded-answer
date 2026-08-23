# The Grounded Answer

**Hackathon:** Brite Spark 2026  
**Problem:** Problem 1 — The Grounded Answer  
**Category:** AI / RAG

## Description

The Grounded Answer is a CLI-based policy question-answering assistant. Given a plain-language question, it retrieves relevant clauses from a supplied policy manual, evaluates whether the evidence actually supports a definitive answer, and either produces a grounded, cited response or explicitly refuses when the policy does not settle the matter. Every substantive claim in an answer is traceable to a specific policy clause. The system deliberately separates retrieval, evidence sufficiency evaluation, and answer generation to avoid the dangerous failure mode of a fluent-but-unsupported response.

## Current Status

**Milestone 3 — Clause-Level Parsing and Structured Clause Store** ✅  
**Milestone 2 — Markdown Policy Ingestion** ✅  
**Milestone 1 — Project Foundation** ✅

The policy corpus is fully parsed into 137 structured, citable clauses.
The RAG pipeline (retrieval, evidence evaluation, answer generation, citations)
is not yet implemented.

## Local Setup

### Prerequisites

- Python 3.11
- An OpenAI API key (for future milestones)

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

# Ask a question (pipeline not yet implemented)
python -m src ask "What is the resource limit?"
```

## Running the Smoke Tests

```bash
pytest
```

Expected output: all 53 tests pass.

## Ingestion API

The ingestion layer is in `src/ingestion/`.

```python
from src.ingestion import load_policy_document, inspect_markdown
from src.ingestion import parse_clauses, ClauseStore

# Load the policy manual
doc = load_policy_document("data/raw/policy_manual.md")

# Inspect its structure (optional)
insp = inspect_markdown(doc)
print(insp.heading_counts_by_level)   # {1: 15, 2: 54}

# Parse all 137 authoritative clauses
clauses = parse_clauses(doc)
store = ClauseStore(clauses)
print(store.count())                   # 137

# Retrieve a specific clause
c = store.get_by_id("4.3.2")
print(c.clause_id)                     # "4.3.2"
print(c.part_id)                       # "4"
print(c.section_id)                    # "4.3"
print(c.cross_references)             # ()
print(c.text)
# **4.3.2** A recipient must report any change in household
# composition … within **10 calendar days** …

# Each clause carries its source location
print(c.start_line, c.end_line)        # 200  200

# All clauses in document order
for clause in store.all():
    print(f"§{clause.clause_id}  {clause.text[:60]}")
```

## Project Structure

```
src/
  app.py            # CLI entry point (Typer)
  ingestion/
    loader.py       # load_policy_document() → PolicyDocument
    inspector.py    # inspect_markdown() → MarkdownInspection
    parser.py       # parse_clauses() → list[PolicyClause]
    store.py        # ClauseStore — in-memory clause index
  retrieval/        # Hybrid retrieval (semantic + BM25) — not yet implemented
  evidence/         # Evidence sufficiency evaluation — not yet implemented
  generation/       # Grounded answer construction — not yet implemented
  citation/         # Deterministic citation rendering — not yet implemented
  models/           # Shared Pydantic schemas — not yet implemented
data/
  raw/              # Source policy document (policy_manual.md)
  processed/        # Parsed clause store (JSON) — not yet implemented
evaluation/         # Ten-question evaluation set — not yet implemented
tests/              # pytest test suite
```

## Architecture Overview

```
Question
  → Hybrid Retrieval (Semantic + BM25)
  → Candidate Clauses
  → Evidence Evaluation
      ├── SUPPORTED    → Generate grounded answer + citations
      ├── INSUFFICIENT → Explicit refusal + next action
      └── CONFLICTING  → Surface both clauses, explain conflict
```
