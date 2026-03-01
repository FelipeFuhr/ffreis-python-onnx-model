#!/usr/bin/env python3
"""Main application entry point."""

from pathlib import Path
from sys import path as sys_path

# Support running `python main.py` from repository root without installation.
sys_path.insert(0, str(Path(__file__).resolve().parent / "src"))

from onnx_model_serving.lib import greet  # noqa: E402


def main() -> None:
    """Run the application entry point."""
    print(greet())


if __name__ == "__main__":
    main()
