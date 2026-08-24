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
DEFAULT_AMENDMENT = Path("data/raw/Amendment No. 2026-01.md")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The policy question to answer."),
    corpus: Path = typer.Option(DEFAULT_CORPUS, help="Path to policy manual Markdown file."),
    amendment: Path = typer.Option(DEFAULT_AMENDMENT, help="Path to amendment Markdown file if applicable."),
    date: str | None = typer.Option(None, "--date", "-d", help="Relevant claim or determination date (YYYY-MM-DD) to evaluate policy applicability."),
) -> None:
    """Ask a question against the policy manual and get a grounded, cited answer."""
    from datetime import date as dt_date
    from src.pipeline import PolicyQAPipeline

    if not corpus.exists():
        typer.echo(f"Error: Policy corpus not found at {corpus}", err=True)
        raise typer.Exit(code=1)

    parsed_date = None
    if date is not None:
        try:
            parsed_date = dt_date.fromisoformat(date.strip())
        except ValueError:
            typer.echo(f"Error: Invalid date format {date!r}. Expected YYYY-MM-DD (e.g. 2026-02-20).", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Question: {question}")
    if parsed_date:
        typer.echo(f"Date: {parsed_date.isoformat()}")
    typer.echo("")

    amendment_path = amendment if amendment.exists() else None
    pipeline = PolicyQAPipeline.build_from_corpus(
        corpus_path=corpus,
        amendment_path=amendment_path,
    )
    answer = pipeline.ask(question, date=parsed_date)

    # Output formatted response
    typer.echo(f"Status: [{answer.status.value}]")
    typer.echo(f"Answer: {answer.answer_text}\n")

    if answer.verifiable_citations:
        typer.echo("Citations:")
        for cit in answer.verifiable_citations:
            source = cit.source_label.removesuffix(".md")
            typer.echo(f"  - §{cit.clause_id}, {cit.line_label}")
            typer.echo(f"    Source: {source}")
            typer.echo(f"    {cit.source_url}")
    elif answer.citations:
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
    typer.echo("Status: Day-1 Milestone 6 + Day-2 Milestones 1-5 (Verifiable Citations) complete.")
    typer.echo("Pipeline: Ingestion → Hybrid Retrieval → Temporal Filter → Evidence Evaluation → Grounded Answer Generation")


if __name__ == "__main__":
    app()

