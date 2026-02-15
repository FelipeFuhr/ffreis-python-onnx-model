"""Library functions for the application."""


def add(a: int, b: int) -> int:
    """Add two integers.

    Parameters
    ----------
    a : int
        First value.
    b : int
        Second value.

    Returns
    -------
    int
        Sum of ``a`` and ``b``.
    """
    return a + b


def greet() -> str:
    """Return a greeting message.

    Returns
    -------
    str
        Greeting text.
    """
    return "Hello, world!"


def is_even(n: int) -> bool:
    """Check whether a number is even.

    Parameters
    ----------
    n : int
        Number to evaluate.

    Returns
    -------
    bool
        ``True`` if ``n`` is even, otherwise ``False``.
    """
    return n % 2 == 0


def clamp(n: int, minimum: int, maximum: int) -> int:
    """Clamp a value between lower and upper bounds.

    Parameters
    ----------
    n : int
        Value to clamp.
    minimum : int
        Lower inclusive bound.
    maximum : int
        Upper inclusive bound.

    Returns
    -------
    int
        Clamped value.
    """
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n


def repeat_word(word: str, times: int) -> str:
    """Repeat a word a number of times.

    Parameters
    ----------
    word : str
        Token to repeat.
    times : int
        Number of repetitions.

    Returns
    -------
    str
        Repeated tokens joined by spaces.
    """
    return " ".join([word] * times)


def sum_list(nums: list[int]) -> int:
    """Sum a list of integers.

    Parameters
    ----------
    nums : list[int]
        Integer values to sum.

    Returns
    -------
    int
        Sum of all values.
    """
    return sum(nums)


def first_non_empty(values: list[str]) -> str | None:
    """Return the first non-empty string.

    Parameters
    ----------
    values : list[str]
        Candidate string values.

    Returns
    -------
    str | None
        First non-empty value, or ``None`` when no non-empty entry exists.
    """
    for value in values:
        if value:
            return value
    return None


def toggle(flag: bool) -> bool:
    """Invert a boolean value.

    Parameters
    ----------
    flag : bool
        Input boolean value.

    Returns
    -------
    bool
        Inverted boolean value.
    """
    return not flag
