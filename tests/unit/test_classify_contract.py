"""Приём ответа модели: что считается пригодным элементом."""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.config import ModelConfig
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.llm.classify import CategoryClassifier, TransactionInput
from halyk.llm.runner import Budget
from halyk.models.classification import TransactionCategory as C

ROW = TransactionInput(
    txn_id="TXN-P1-0001",
    description="Quarterly interest coupon on term loan",
    counterparty="Bank",
    amount="-100.00",
    currency="USD",
    date="2025-06-01",
)


def item(**overrides: Any) -> dict[str, Any]:
    base = {
        "txn_id": ROW.txn_id,
        "category": "interest_expense",
        "confidence": 0.9,
        "evidence": "Quarterly interest coupon",
    }
    return base | overrides


def verdicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    return CategoryClassifier._verdicts(items, [ROW])


def test_valid_item_is_accepted() -> None:
    found = verdicts([item()])
    assert found[ROW.txn_id].category is C.INTEREST_EXPENSE


def test_evidence_must_come_from_the_description() -> None:
    """Придуманное обоснование означает придуманную статью, и по confidence это не видно."""
    assert verdicts([item(evidence="это явно проценты по кредиту")]) == {}


def test_evidence_is_matched_ignoring_case_and_spacing() -> None:
    assert verdicts([item(evidence="  QUARTERLY   INTEREST coupon ")])


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_outside_the_range_is_rejected(confidence: float) -> None:
    assert verdicts([item(confidence=confidence)]) == {}


def test_empty_evidence_is_rejected() -> None:
    assert verdicts([item(evidence="")]) == {}


def test_unknown_category_is_rejected() -> None:
    assert verdicts([item(category="прочее")]) == {}


def test_foreign_identifier_is_rejected() -> None:
    assert verdicts([item(txn_id="TXN-P9-9999")]) == {}


def test_duplicate_identifier_keeps_only_the_first() -> None:
    found = verdicts([item(), item(category="opex")])
    assert found[ROW.txn_id].category is C.INTEREST_EXPENSE


def test_invalid_item_becomes_a_missing_row(tmp_path: Path) -> None:
    """Непригодный элемент — это не ответ, и строка уходит в повтор."""

    class Engine(CategoryClassifier):
        def __init__(self) -> None:
            super().__init__(
                config=ModelConfig(name="gpt-5.6-sol", api_key="test"),
                cache=ModelCache(directory=tmp_path / "c", policy=CachePolicy.READ_WRITE),
            )
            self.asked: list[tuple[str, ...]] = []

        def _ask(self, rows: Any) -> tuple[dict[str, Any], tuple[None, None, None], str]:
            self.asked.append(tuple(r.txn_id for r in rows))
            broken = len(self.asked) == 1
            return (
                {"items": [item(evidence="выдумка" if broken else "interest coupon")]},
                (None, None, None),
                "req",
            )

    engine = Engine()
    found = engine.classify("ACC-7801", [ROW])
    assert engine.asked == [(ROW.txn_id,), (ROW.txn_id,)]
    assert engine.calls[0].missing == (ROW.txn_id,)
    assert found[ROW.txn_id].category is C.INTEREST_EXPENSE


def test_classifier_charges_its_own_model_price(tmp_path: Path) -> None:
    class Engine(CategoryClassifier):
        def _ask(self, rows: Any) -> tuple[dict[str, Any], tuple[None, None, None], str]:
            return {"items": [item()]}, (None, None, None), "req"

    budget = Budget(
        price_input_per_million=Decimal(5),
        price_output_per_million=Decimal(30),
    )
    engine = Engine(
        config=ModelConfig(
            name="gpt-5.6-luna",
            api_key="test",
            price_input_per_million=Decimal("0.2"),
            price_output_per_million=Decimal("1.2"),
        ),
        cache=ModelCache(directory=tmp_path / "c", policy=CachePolicy.READ_WRITE),
        budget=budget,
    )

    expected = Decimal(engine._estimate([ROW])) * Decimal("0.2") / Decimal(1_000_000)
    engine.classify("ACC-7801", [ROW])

    assert budget.estimated_cost == expected
