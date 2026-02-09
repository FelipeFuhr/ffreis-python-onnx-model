"""Tests for lib.py functions."""

from lib import (
    add,
    greet,
    is_even,
    clamp,
    repeat_word,
    sum_list,
    first_non_empty,
    toggle,
)


def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_greet():
    """Test greeting function."""
    assert greet() == "Hello, world!"


def test_is_even():
    """Test even number check."""
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True
    assert is_even(-2) is True


def test_clamp():
    """Test clamp function."""
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_repeat_word():
    """Test repeat word function."""
    assert repeat_word("hello", 3) == "hello hello hello"
    assert repeat_word("test", 1) == "test"
    assert repeat_word("x", 0) == ""


def test_sum_list():
    """Test sum list function."""
    assert sum_list([1, 2, 3, 4, 5]) == 15
    assert sum_list([]) == 0
    assert sum_list([-1, 1]) == 0


def test_first_non_empty():
    """Test first non-empty string finder."""
    assert first_non_empty(["", "hello", "world"]) == "hello"
    assert first_non_empty(["test"]) == "test"
    assert first_non_empty(["", "", ""]) is None
    assert first_non_empty([]) is None


def test_toggle():
    """Test boolean toggle."""
    assert toggle(True) is False
    assert toggle(False) is True
