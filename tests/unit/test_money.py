from decimal import Decimal

import pytest

from halyk.money import (
    Currency,
    CurrencyMismatchError,
    Money,
    MoneyParseError,
    Rounding,
    detect_currency,
    parse_money,
    quantize,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$7,100,000.00", 710_000_000),
        ("$450,000.00", 45_000_000),
        ("$1,234.56", 123_456),
        ("$1,234", 123_400),
        ("50000000", 5_000_000_000),
        ("50 000 000", 5_000_000_000),
        ("50 000 000,00", 5_000_000_000),
        ("50 000 000.50", 5_000_000_050),
        ("1 234,5", 123_450),
        ("50 млн", 5_000_000_000),
        ("50 тыс.", 5_000_000),
        ("-1 000", -100_000),
    ],
)
def test_parse_money(text: str, expected: int | None) -> None:
    if expected is None:
        with pytest.raises(MoneyParseError):
            parse_money(text, Currency.USD)
    else:
        assert parse_money(text, Currency.USD).minor == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$3.75 million", "3750000.00"),
        ("3,75 млн USD", "3750000.00"),
        ("1,5 млрд", "1500000000.00"),
        ("2.5 thousand", "2500.00"),
        ("1,250 млн", "1250000000.00"),
    ],
)
def test_fractional_multipliers(text: str, expected: str) -> None:
    """Дробная часть с множителем однозначна и должна считаться, а не падать.

    Разрядный разделитель отделяет ровно три цифры, десятичный — одну-две, поэтому
    «3,75 млн» и «1,250 млн» читаются по-разному и оба верно.
    """
    assert parse_money(text, Currency.USD).to_decimal() == Decimal(expected)


def test_thousands_separator_is_not_a_decimal_point() -> None:
    # Порог «$1,500,000.00» из договора и «1,50» — разные числа при одном символе.
    assert parse_money("$1,500,000.00").to_decimal() == Decimal("1500000.00")
    assert parse_money("1,50", Currency.USD).to_decimal() == Decimal("1.50")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$450,000.00", Currency.USD),
        ("100 EUR", Currency.EUR),
        ("50 млн тенге", Currency.KZT),
        ("450 000", None),
    ],
)
def test_detect_currency(text: str, expected: Currency | None) -> None:
    assert detect_currency(text) is expected


def test_currency_comes_from_text_before_argument() -> None:
    assert parse_money("€1,000.00", Currency.USD).currency is Currency.EUR


def test_currency_is_required() -> None:
    with pytest.raises(MoneyParseError):
        parse_money("450 000")


def test_words_in_brackets_are_ignored() -> None:
    # Пропись в скобках нужна для сверки человеком, а не для расчёта.
    assert parse_money("50 000 000 (пятьдесят миллионов) тенге").minor == 5_000_000_000


def test_neighbouring_number_does_not_leak_into_multiplier() -> None:
    assert parse_money("не менее 500 000 тенге по счёту 20 млн").minor == 50_000_000


def test_no_number() -> None:
    with pytest.raises(MoneyParseError):
        parse_money("сумма не определена", Currency.USD)


def test_roundtrip_keeps_cents() -> None:
    assert Money.from_decimal("1234.56", Currency.USD).to_decimal() == Decimal("1234.56")


def test_fraction_of_minor_unit_is_rejected() -> None:
    with pytest.raises(MoneyParseError):
        Money.from_decimal("0.001", Currency.USD)


def test_boundary_sum_is_exact() -> None:
    # Тот самый случай, ради которого деньги целые: сумма ровно на пороге.
    rows = [Money.from_decimal("0.10", Currency.USD)] * 3
    total = sum(rows[1:], rows[0])
    assert total == Money.from_decimal("0.30", Currency.USD)


def test_currencies_do_not_mix() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money.from_decimal("1", Currency.USD) + Money.from_decimal("1", Currency.EUR)


def test_abs_and_negation_keep_currency() -> None:
    spend = Money.from_decimal("-283664.18", Currency.USD)
    assert abs(spend) == Money.from_decimal("283664.18", Currency.USD)
    assert (-spend).currency is Currency.USD


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
