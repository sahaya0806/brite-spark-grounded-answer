"""
Allows the package to be run as:
    python -m src ask "..."
    python -m src info
"""
from src.app import app

if __name__ == "__main__":
    app()
