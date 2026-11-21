"""Pytest configuration.

Ensures the repository root is on ``sys.path`` so that ``import src...`` works
when running the test suite from any working directory. The project is laid out
as implicit namespace packages (no ``__init__.py`` under ``src``), matching how
``demo.py`` imports modules as ``src.<module>``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
