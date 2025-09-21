"""Test fixtures and path configuration."""

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the project ``src`` directory is importable during tests."""

    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
