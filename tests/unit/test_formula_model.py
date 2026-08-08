"""AST должен выражать те пункты, которые реально встретились в открытом наборе.

Тесты здесь не считают чисел — исполнителя ещё нет. Они проверяют выразительность:
если форма пункта на этом языке не записывается, дальше идти незачем.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.formula import (
    Condition,
    Constant,
    CovenantFormula,
    Difference,
    Direction,
    EvidenceMode,
    ExternalMetric,
    FactValue,
    Largest,
    LedgerSum,
    Ratio,
    RelatedParty,
    Selector,
    Sum,
)
from halyk.models.source import SourceRef

SOURCE = SourceRef(file_hash="a" * 64, file_name="04eee2e9ba8c.pdf", page=3)


def formula(**overrides: object) -> CovenantFormula:
    base: dict[str, object] = {
        "scenario_id": "P1",
        "clause_id": "6.1",
        "title": "проверка",
        "measure": LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
        "operator": Operator.LE,
        "threshold": Constant(value=Decimal("300000")),
        "unit": Unit.MONEY,
        "source_refs": (SOURCE,),
        "quote": "не более 300 000 долларов",
        "confidence": 0.9,
    }
    return CovenantFormula(**(base | overrides))  # type: ignore[arg-type]


def spend(*categories: Cat) -> LedgerSum:
    return LedgerSum(selector=Selector(categories=categories, direction=Direction.OUTFLOW))


def test_capex_against_opex_and_rent() -> None:
    """P1/6.1: капитальные затраты против суммы операционных расходов и аренды."""
    spec = formula(
        measure=Ratio(numerator=spend(Cat.CAPEX), denominator=spend(Cat.OPEX, Cat.RENT)),
        unit=Unit.RATIO,
        threshold=Constant(value=Decimal("0.045")),
    )
    assert spec.measure.op == "div"


def test_interest_coverage_with_add_backs() -> None:
    """B1/6.1: EBITDA с добавлением разовых статей, делённая на процентные расходы."""
    ebitda = Sum(
        terms=(
            Difference(
                left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                right=spend(Cat.OPEX),
            ),
            FactValue(fact_kind="one_off_item", above_one_off_policy=True),
        )
    )
    spec = formula(
        measure=Ratio(numerator=ebitda, denominator=spend(Cat.INTEREST_EXPENSE)),
        unit=Unit.RATIO,
        operator=Operator.GE,
        threshold=Constant(value=Decimal("1.50")),
        evidence=EvidenceMode.NONE,
    )
    assert spec.evidence is EvidenceMode.NONE


def test_two_caps_measured_separately() -> None:
    """B1/6.2: оплата труда и коммунальные — по отдельности, а не в совокупности."""
    spec = formula(
        measure=Largest(values=(spend(Cat.PAYROLL), spend(Cat.UTILITIES))),
        threshold=Constant(value=Decimal("1500000")),
    )
    assert len(spec.measure.values) == 2


def test_fourth_quarter_revenue() -> None:
    """Квартальный пункт режет период месяцами, а не сдвигает его границы."""
    spec = formula(
        measure=LedgerSum(
            selector=Selector(categories=(Cat.REVENUE,), months=(10, 11, 12)),
        ),
        operator=Operator.GE,
    )
    assert spec.measure.selector.months == (10, 11, 12)


def test_related_party_share() -> None:
    """Доля расчётов со связанными сторонами в выручке."""
    spec = formula(
        measure=Ratio(
            numerator=LedgerSum(
                selector=Selector(categories=(Cat.REVENUE,), related_party=RelatedParty.ONLY)
            ),
            denominator=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
        ),
        unit=Unit.RATIO,
    )
    assert spec.measure.numerator.selector.related_party is RelatedParty.ONLY


def test_springing_covenant_keeps_its_measure() -> None:
    """Пункт со срабатыванием: условие отдельно, измеряемая величина всё равно есть."""
    spec = formula(
        applies_when=Condition(
            left=spend(Cat.CAPEX),
            operator=Operator.GT,
            right=Constant(value=Decimal("1000000")),
        )
    )
    assert spec.applies_when is not None
    assert spec.measure is not None


def test_external_metric_is_expressible() -> None:
    """P5: групповые капзатраты, которых в датасете нет, — узел, а не подставленное число."""
    spec = formula(
        measure=Ratio(
            numerator=spend(Cat.CAPEX),
            denominator=ExternalMetric(
                name="group_capex", description="капзатраты группы, в наборе не раскрыты"
            ),
        ),
        unit=Unit.RATIO,
    )
    assert spec.measure.denominator.op == "external"


def test_open_selector_is_visible() -> None:
    """Селектор без ограничений валидацию проходит, но обязан себя объявлять."""
    assert Selector().is_open()
    assert not Selector(categories=(Cat.OPEX,)).is_open()


def test_unknown_node_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CovenantFormula.model_validate(
            {
                "scenario_id": "P1",
                "clause_id": "6.1",
                "title": "проверка",
                "measure": {"op": "sqrt", "value": 1},
                "operator": "le",
                "threshold": {"op": "constant", "value": "1"},
                "unit": "money",
                "source_refs": [SOURCE.model_dump(mode="json")],
                "quote": "…",
                "confidence": 0.5,
            }
        )


def test_binary_nodes_demand_two_operands() -> None:
    with pytest.raises(ValidationError):
        Sum(terms=(Constant(value=Decimal(1)),))


def test_round_trip_through_json() -> None:
    """Дерево переживает сериализацию: иначе его не положить в артефакт прогона."""
    spec = formula(
        measure=Ratio(numerator=spend(Cat.CAPEX), denominator=spend(Cat.OPEX, Cat.RENT)),
        unit=Unit.RATIO,
    )
    restored = CovenantFormula.model_validate_json(spec.model_dump_json())
    assert restored == spec
