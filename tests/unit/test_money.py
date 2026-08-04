from decimal import Decimal

import pytest

from halyk.money import (
    MoneyParseError,
    Rounding,
    parse_amount,
    quantize,
    tenge_to_tiyn,
    tiyn_to_tenge,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("50000000", 5_000_000_000),
        ("50 000 000", 5_000_000_000),
        ("50 000 000,00", 5_000_000_000),
        ("50 000 000.50", 5_000_000_050),
        ("1 234,5", 123_450),
        ("50 млн тенге", 5_000_000_000),
        ("50 тыс. тенге", 5_000_000),
        ("1,5 млрд", None),
        ("-1 000", -100_000),
    ],
)
def test_parse_amount(text: str, expected: int | None) -> None:
    if expected is None:
        with pytest.raises(MoneyParseError):
            parse_amount(text)
    else:
        assert parse_amount(text) == expected


def test_words_in_brackets_are_ignored() -> None:
    # Пропись в скобках нужна для сверки человеком, а не для расчёта.
    assert parse_amount("50 000 000 (пятьдесят миллионов) тенге") == 5_000_000_000


def test_neighbouring_number_does_not_leak_into_multiplier() -> None:
    assert parse_amount("не менее 500 000 тенге по счёту 20 млн") == 50_000_000


def test_no_number() -> None:
    with pytest.raises(MoneyParseError):
        parse_amount("сумма не определена")


def test_roundtrip_keeps_kopecks() -> None:
    assert tiyn_to_tenge(tenge_to_tiyn("1234.56")) == Decimal("1234.56")


def test_fraction_of_tiyn_is_rejected() -> None:
    with pytest.raises(MoneyParseError):
        tenge_to_tiyn("0.001")


def test_boundary_sum_is_exact() -> None:
    # Тот самый случай, ради которого деньги целые: сумма ровно на пороге.
    rows = [tenge_to_tiyn("0.10")] * 3
    assert sum(rows) == tenge_to_tiyn("0.30")


@pytest.mark.parametrize(
    ("value", "mode", "expected"),
    [
        ("2.345", Rounding.HALF_UP, "2.35"),
        ("2.345", Rounding.HALF_EVEN, "2.34"),
        ("2.355", Rounding.HALF_EVEN, "2.36"),
    ],
)
def test_quantize(value: str, mode: Rounding, expected: str) -> None:
    assert quantize(Decimal(value), scale=2, mode=mode) == Decimal(expected)
