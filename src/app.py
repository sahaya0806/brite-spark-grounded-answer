"""
The Grounded Answer — CLI entry point.

Usage:
    python -m src ask "Your question here"
    python -m src info
"""

from __future__ import annotations

from pathlib import Path
import typer
from dotenv import load_dotenv

# Load .env variables if present
load_dotenv()

app = typer.Typer(
    name="grounded-answer",
    help="Policy question answering with grounded, cited answers.",
    add_completion=False,
)

DEFAULT_CORPUS = Path("data/raw/policy_manual.md")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The policy question to answer."),
    corpus: Path = typer.Option(DEFAULT_CORPUS, help="Path to policy manual Markdown file."),
) -> None:
    """Ask a question against the policy manual and get a grounded, cited answer."""
    from src.pipeline import PolicyQAPipeline

    if not corpus.exists():
        typer.echo(f"Error: Policy corpus not found at {corpus}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Question: {question}\n")

    pipeline = PolicyQAPipeline.build_from_corpus(corpus)
    answer = pipeline.ask(question)

    # Output formatted response
    typer.echo(f"Status: [{answer.status.value}]")
    typer.echo(f"Answer: {answer.answer_text}\n")

    if answer.citations:
        typer.echo("Citations:")
        for cit in answer.citations:
            typer.echo(f"  - {cit}")
    else:
        typer.echo("Citations: None")


@app.command()
def info() -> None:
    """Display current system status."""
    typer.echo("The Grounded Answer — Brite Spark 2026")
    typer.echo("Problem 1: The Grounded Answer (AI / RAG)")
    typer.echo("Status: Milestone 6 — Grounded Answer Generation complete.")
    typer.echo("Pipeline: Ingestion → Hybrid Retrieval → Evidence Evaluation → Grounded Answer Generation")


if __name__ == "__main__":
    app()
