from typing import TypeVar
import pytest

T = TypeVar("T", int, float)


def add(a: T, b: T) -> T:
    return a + b


def divide(a: int | float, b: int | float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def test_add_integers() -> None:
    result: int = add(2, 3)
    assert result == 5


def test_add_floats() -> None:
    result: float = add(1.5, 2.5)
    assert result == 4.0


def test_divide() -> None:
    result: float = divide(10, 2)
    assert result == 5.0


def test_divide_by_zero() -> None:
    with pytest.raises(ValueError):
        divide(10, 0)
