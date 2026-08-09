"""Детерминированные правила отнесения к статье."""

import pytest

from halyk.knowledge.rules import Direction, classify_by_rules
from halyk.models.classification import TransactionCategory as C


def category(description: str, minor: int) -> C | None:
    match = classify_by_rules(description, minor)
    return match.category if match else None


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Quarterly interest coupon", C.INTEREST_EXPENSE),
        ("Default interest charge", C.INTEREST_EXPENSE),
        ("Corporate income tax instalment", C.TAXES),
        ("Monthly payroll disbursement", C.PAYROLL),
        ("Electricity network capacity charge", C.UTILITIES),
        ("Office rent", C.RENT),
        ("Property insurance premium", C.INSURANCE_PREMIUM),
        ("Equipment purchase for the kiln", C.CAPEX),
        ("Plant operating and maintenance expenses", C.OPEX),
        ("Kiln servicing and operating costs 2025", C.OPEX),
    ],
)
def test_outflows_are_recognised(description: str, expected: C) -> None:
    assert category(description, -100_00) is expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Sales settlement", C.REVENUE),
        ("Bridge loan proceeds received", C.FINANCING_INFLOW),
        ("Interest income on treasury bills", C.OTHER),
        ("Interest credited on current account", C.OTHER),
    ],
)
def test_inflows_are_recognised(description: str, expected: C) -> None:
    assert category(description, 100_00) is expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Tax overpayment refunded", C.TAXES),
        ("Excise tax credit received", C.TAXES),
        ("Rent deposit returned", C.RENT),
        ("Lease incentive received", C.RENT),
        ("Insurance broker rebate", C.INSURANCE_PREMIUM),
        ("Group insurance experience refund", C.INSURANCE_PREMIUM),
        ("Payroll overfunding returned", C.PAYROLL),
        ("Telecom service credit received", C.UTILITIES),
        ("Marketing overbilling refund", C.MARKETING),
    ],
)
def test_returns_keep_the_article_of_the_original_expense(description: str, expected: C) -> None:
    """Возврат относится к той же статье, но остаётся поступлением.

    Статью определяет существо операции, а знак решает, войдёт ли она в сумму
    расходов, — это решает формула ковенанта, а не классификатор.
    """
    assert category(description, 100_00) is expected


def test_the_same_word_gives_different_articles_by_direction() -> None:
    """«interest» — процентный расход только когда это платёж."""
    assert category("Interest on subordinated notes", -100_00) is C.INTEREST_EXPENSE
    assert category("Interest income on treasury bills", 100_00) is C.OTHER


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Customer newsletter marketing production", C.MARKETING),
        ("Management advisory retainer", C.OTHER),
        ("Outdoor marketing site hire", C.MARKETING),
        ("Industry exhibition stand marketing", C.MARKETING),
        ("Statutory audit fee 2025", C.OTHER),
    ],
)
def test_administrative_costs_are_not_operating_expenses(description: str, expected: C) -> None:
    """Строка «операционные расходы» договора — про эксплуатацию объекта.

    Маркетинг, реклама и консультации операционные по смыслу, но в эту строку
    отчётности не входят: у B1 она состоит из одной проводки на шесть миллионов, а
    рекламы там на двадцать. Маркетинг при этом держится отдельной статьёй: его
    потолок ковенанты задают прямо, и внутри «прочего» такую величину не выразить.
    """
    assert category(description, -100_00) is expected


def test_unknown_description_is_left_to_the_model() -> None:
    assert classify_by_rules("Berth silt cleaning and clearance works", -100_00) is None
    assert classify_by_rules("Zhaiyk dredging arrangement", -100_00) is None


def test_direction_of_a_missing_amount_matches_any_rule() -> None:
    """У строки без суммы направление неизвестно, и правило по знаку не отсекается."""
    assert Direction.of(None) is Direction.ANY
    assert category("Corporate income tax instalment", 0) is C.TAXES
