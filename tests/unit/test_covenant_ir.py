from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from halyk.models.covenant import (
    Aggregation,
    Comparison,
    CovenantIR,
    EvidenceRule,
    Operator,
    Period,
    Unit,
)
from halyk.models.source import SourceRef

SOURCE = SourceRef(file_hash="0" * 64, file_name="dogovor.pdf", page=4)


def build(**overrides: object) -> CovenantIR:
    base: dict[str, object] = {
        "borrower_id": "B-001",
        "covenant_id": "C-001",
        "clause_id": "5.2",
        "metric": "credit_turnover",
        "measurement_period": Period(start=date(2026, 1, 1), end=date(2026, 3, 31)),
        "aggregation": Aggregation.SUM,
        "comparison": Comparison(
            operator=Operator.GE,
            threshold=Decimal("50000000"),
            unit=Unit.MONEY,
            currency="KZT",
        ),
        "evidence_rule": EvidenceRule.FIRST_VIOLATION,
        "source_refs": (SOURCE,),
        "confidence": 0.9,
    }
    return CovenantIR(**(base | overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operator", "measured", "holds"),
    [
        (Operator.GE, Decimal("50000000"), True),
        (Operator.GT, Decimal("50000000"), False),
        (Operator.LE, Decimal("49999999.99"), True),
        (Operator.LT, Decimal("50000000"), False),
    ],
)
def test_operator_on_boundary(operator: Operator, measured: Decimal, holds: bool) -> None:
    assert operator.holds(measured, Decimal("50000000")) is holds


def test_money_threshold_requires_currency() -> None:
    with pytest.raises(ValidationError):
        Comparison(operator=Operator.GE, threshold=Decimal(1), unit=Unit.MONEY)


def test_period_order_is_checked() -> None:
    with pytest.raises(ValidationError):
        Period(start=date(2026, 3, 31), end=date(2026, 1, 1))


def test_aggregate_covenant_cannot_point_at_single_row() -> None:
    # У среднего остатка нет одной виноватой транзакции, см. docs/adr/0005.
    with pytest.raises(ValidationError):
        build(aggregation=Aggregation.BALANCE_AVERAGE)


def test_aggregate_covenant_with_contributor_rule() -> None:
    ir = build(
        aggregation=Aggregation.BALANCE_AVERAGE,
        evidence_rule=EvidenceRule.LARGEST_CONTRIBUTOR,
    )
    assert ir.aggregation is Aggregation.BALANCE_AVERAGE


def test_ir_is_frozen() -> None:
    ir = build()
    with pytest.raises(ValidationError):
        ir.confidence = 0.1  # type: ignore[misc]
