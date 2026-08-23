"""
The Grounded Answer — CLI entry point.

Usage:
    python -m src.app ask "Your question here"
    python -m src.app info

This module will grow to support the full RAG pipeline.
For now it demonstrates that the application foundation starts cleanly.
"""

import typer

app = typer.Typer(
    name="grounded-answer",
    help="Policy question answering with grounded, cited answers.",
    add_completion=False,
)


@app.command()
def ask(question: str = typer.Argument(..., help="The policy question to answer.")) -> None:
    """Ask a question against the policy manual."""
    typer.echo("The Grounded Answer — policy assistant")
    typer.echo(f"Question: {question}")
    typer.echo(
        "\n[Pipeline not yet implemented — see upcoming milestones.]\n"
        "Ingestion → Retrieval → Evidence Evaluation → Answer / Refuse / Conflict"
    )


@app.command()
def info() -> None:
    """Display current system status."""
    typer.echo("The Grounded Answer — Brite Spark 2026")
    typer.echo("Problem 1: The Grounded Answer (AI / RAG)")
    typer.echo("Status: Milestone 1 — Project foundation complete.")
    typer.echo("Pipeline: not yet implemented.")


if __name__ == "__main__":
    app()
