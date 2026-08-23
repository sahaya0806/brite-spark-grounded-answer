"""
Smoke tests for Milestone 1 — Project Foundation.

These tests verify that:
- All source packages can be imported without errors.
- The CLI application object is correctly initialised.
- The package structure is in place.

They do NOT test any RAG functionality (not yet implemented).
"""

import importlib


def test_src_package_importable():
    """The top-level src package must be importable."""
    mod = importlib.import_module("src")
    assert mod is not None


def test_ingestion_package_importable():
    """src.ingestion must be importable."""
    mod = importlib.import_module("src.ingestion")
    assert mod is not None


def test_retrieval_package_importable():
    """src.retrieval must be importable."""
    mod = importlib.import_module("src.retrieval")
    assert mod is not None


def test_evidence_package_importable():
    """src.evidence must be importable."""
    mod = importlib.import_module("src.evidence")
    assert mod is not None


def test_generation_package_importable():
    """src.generation must be importable."""
    mod = importlib.import_module("src.generation")
    assert mod is not None


def test_citation_package_importable():
    """src.citation must be importable."""
    mod = importlib.import_module("src.citation")
    assert mod is not None


def test_models_package_importable():
    """src.models must be importable."""
    mod = importlib.import_module("src.models")
    assert mod is not None


def test_app_importable():
    """src.app must be importable and expose a Typer app object."""
    mod = importlib.import_module("src.app")
    assert hasattr(mod, "app"), "src.app must expose a 'app' Typer instance"


def test_cli_app_is_typer_instance():
    """The CLI app object must be a Typer instance."""
    import typer
    from src.app import app as cli_app
    assert isinstance(cli_app, typer.Typer)


def test_evaluation_package_importable():
    """evaluation package must be importable."""
    mod = importlib.import_module("evaluation")
    assert mod is not None
