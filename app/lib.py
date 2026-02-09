"""Library functions for the application."""


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def greet() -> str:
    """Return a greeting message."""
    return "Hello, world!"


def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0


def clamp(n: int, minimum: int, maximum: int) -> int:
    """Clamp a value between min and max."""
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n


def repeat_word(word: str, times: int) -> str:
    """Repeat a word a number of times."""
    return " ".join([word] * times)


def sum_list(nums: list[int]) -> int:
    """Sum a list of numbers."""
    return sum(nums)


def first_non_empty(values: list[str]) -> str | None:
    """Find the first non-empty string."""
    for value in values:
        if value:
            return value
    return None


def toggle(flag: bool) -> bool:
    """Toggle a boolean value."""
    return not flag
