"""Autodata: Agentic Self-Instruct for synthetic training data (arXiv:2606.25996)."""
from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY from .env into the environment

from .config import Config
from .loop import run_doc
from .trajectory import TrajectoryLogger

__all__ = ["Config", "run_doc", "TrajectoryLogger"]
