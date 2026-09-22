"""Backward-compatible local Streamlit entry point."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from steam_review_rag.app import main  # noqa: E402


main(PROJECT_DIR)
