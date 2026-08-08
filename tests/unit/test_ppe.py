"""Восстановление капитальных затрат из движения основных средств."""

from __future__ import annotations

from decimal import Decimal

import pytest

from halyk.knowledge.ppe import PpeError, parse_ppe_movement

REPORT = """
Note 7 — Property, Plant and Equipment
There were no disposals of property, plant and equipment during the year.
Year ended 2025-12-31
Net book value at the beginning of the year
$148,028,989.69
Depreciation charge for the year
$15,826,229.43
Net book value at the end of the year
$154,050,122.81
"""


def test_additions_close_the_identity() -> None:
    movement = parse_ppe_movement(REPORT)
    assert movement.additions == Decimal("21847362.55")


def test_disposals_shift_the_result() -> None:
    text = REPORT.replace(
        "There were no disposals of property, plant and equipment during the year.",
        "Disposals\n$1,000,000.00",
    )
    assert parse_ppe_movement(text).additions == Decimal("22847362.55")


def test_missing_section_is_refused() -> None:
    with pytest.raises(PpeError, match="нет раздела"):
        parse_ppe_movement("Consolidated statement of financial position")


def test_missing_required_value_is_refused() -> None:
    text = REPORT.replace("Depreciation charge for the year\n$15,826,229.43\n", "")
    with pytest.raises(PpeError, match="обязательная величина"):
        parse_ppe_movement(text)


def test_undisclosed_disposals_are_refused() -> None:
    """Молчание о выбытиях и явный ноль — разные вещи: во втором случае равенство есть."""
    text = REPORT.replace(
        "There were no disposals of property, plant and equipment during the year.", ""
    )
    with pytest.raises(PpeError, match="Выбытия не раскрыты"):
        parse_ppe_movement(text)


@pytest.mark.parametrize(
    "line",
    [
        "Impairment loss\n$500,000.00",
        "Revaluation surplus\n$500,000.00",
        "Acquisition of subsidiaries\n$500,000.00",
        "Exchange differences\n$500,000.00",
        "Transfers\n$500,000.00",
    ],
)
def test_unknown_movement_stops_the_derivation(line: str) -> None:
    """Незнакомое движение меняет тождество, и выводить капзатраты из него нельзя."""
    with pytest.raises(PpeError, match="не учитываем"):
        parse_ppe_movement(REPORT + "\n" + line)


def test_disclosed_additions_must_agree_with_the_derived_ones() -> None:
    with pytest.raises(PpeError, match="не замыкается"):
        parse_ppe_movement(REPORT + "\nAdditions\n$1.00")


def test_disclosed_additions_are_accepted_when_they_agree() -> None:
    movement = parse_ppe_movement(REPORT + "\nAdditions\n$21,847,362.55")
    assert movement.additions == Decimal("21847362.55")
