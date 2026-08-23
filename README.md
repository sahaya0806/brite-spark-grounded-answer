# The Grounded Answer

**Hackathon:** Brite Spark 2026  
**Problem:** Problem 1 — The Grounded Answer  
**Category:** AI / RAG

## Description

The Grounded Answer is a CLI-based policy question-answering assistant. Given a plain-language question, it retrieves relevant clauses from a supplied policy manual, evaluates whether the evidence actually supports a definitive answer, and either produces a grounded, cited response or explicitly refuses when the policy does not settle the matter. Every substantive claim in an answer is traceable to a specific policy clause. The system deliberately separates retrieval, evidence sufficiency evaluation, and answer generation to avoid the dangerous failure mode of a fluent-but-unsupported response.

## Current Status

**Milestone 1 — Project Foundation** ✅  
Project structure, dependencies, CLI entry point, and smoke tests are in place.  
The RAG pipeline (ingestion, retrieval, evidence evaluation, answer generation, citations) is not yet implemented.

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

Expected output: all 10 smoke tests pass.

## Project Structure

```
src/
  app.py            # CLI entry point (Typer)
  ingestion/        # PDF ingestion and clause parsing
  retrieval/        # Hybrid retrieval (semantic + BM25)
  evidence/         # Evidence sufficiency evaluation
  generation/       # Grounded answer construction
  citation/         # Deterministic citation rendering
  models/           # Shared Pydantic schemas
  data/
    raw/            # Source policy document (PDF)
    processed/      # Parsed clause store (JSON)
evaluation/         # Ten-question evaluation set with pass/fail results
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
