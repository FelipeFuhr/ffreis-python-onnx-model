"""Integration tests for the app entrypoint."""

from io import StringIO
from unittest.mock import patch

import pytest

from main import main

pytestmark = pytest.mark.integration


def test_main_prints_greeting() -> None:
    """Ensure the entrypoint emits the expected greeting string."""
    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        main()
        assert fake_stdout.getvalue().strip() == "Hello, world!"
