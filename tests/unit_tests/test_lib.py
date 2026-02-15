"""Tests for lib.py functions."""

import pytest

from onnx_model_serving.lib import (
    add,
    clamp,
    first_non_empty,
    greet,
    is_even,
    repeat_word,
    sum_list,
    toggle,
)

pytestmark = pytest.mark.unit


def test_add() -> None:
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_greet() -> None:
    """Test greeting function."""
    assert greet() == "Hello, world!"


def test_is_even() -> None:
    """Test even number check."""
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True
    assert is_even(-2) is True


def test_clamp() -> None:
    """Test clamp function."""
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_repeat_word() -> None:
    """Test repeat word function."""
    assert repeat_word("hello", 3) == "hello hello hello"
    assert repeat_word("test", 1) == "test"
    assert repeat_word("x", 0) == ""


def test_sum_list() -> None:
    """Test sum list function."""
    assert sum_list([1, 2, 3, 4, 5]) == 15
    assert sum_list([]) == 0
    assert sum_list([-1, 1]) == 0


def test_first_non_empty() -> None:
    """Test first non-empty string finder."""
    assert first_non_empty(["", "hello", "world"]) == "hello"
    assert first_non_empty(["test"]) == "test"
    assert first_non_empty(["", "", ""]) is None
    assert first_non_empty([]) is None


def test_toggle() -> None:
    """Test boolean toggle."""
    assert toggle(True) is False
    assert toggle(False) is True
