"""End-to-end smoke tests for the app executable path."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_main_script_runs() -> None:
    """Run the main script via Python and assert successful output."""
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project_root / "main.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Hello, world!"
