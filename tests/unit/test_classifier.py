"""Разрешение трёх голосов: документ, правило, модель."""

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.config import ModelConfig
from halyk.ingest.normalise import CovenantPeriod, NormalisedLedger, normalise
from halyk.knowledge.classifier import TransactionClassifier
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.llm.classify import CategoryClassifier
from halyk.models.adjustment import (
    Adjustment,
    ReclassifyAdjustment,
    TransactionSelector,
)
from halyk.models.classification import DecisionSource
from halyk.models.classification import TransactionCategory as C
from halyk.models.document import DocumentKind, DocumentStatus
from halyk.models.source import SourceAuthority, SourceRef
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, Money

PERIOD = CovenantPeriod(start=date(2025, 1, 1), end=date(2025, 12, 31))
ACCOUNT = "ACC-7801"

SOURCE = SourceRef(
    file_hash="a" * 64,
    file_name="report.pdf",
    page=2,
    kind=DocumentKind.AUDIT_PROCEDURES,
    status=DocumentStatus.FINAL,
    authority=SourceAuthority.AUTHORITATIVE,
    account_id=ACCOUNT,
)


def row(txn_id: str, description: str, amount: str = "-100.00") -> LedgerRow:
    return LedgerRow(
        txn_id=txn_id,
        date=date(2025, 6, 1),
        account_id=ACCOUNT,
        counterparty="Irtysh Advisory Bureau",
        description=description,
        amount=Money.from_decimal(Decimal(amount), Currency.USD),
    )


def ledger(rows: list[LedgerRow], adjustments: list[Adjustment] | None = None) -> NormalisedLedger:
    return normalise(rows, adjustments or [], PERIOD)


class StubEngine(CategoryClassifier):
    """Движок с заранее известными ответами и настоящим кэшем."""

    def __init__(self, tmp_path: Path, name: str, answers: dict[str, tuple[str, float]]) -> None:
        super().__init__(
            config=ModelConfig(name="gpt-5.6-sol", api_key="test"),
            cache=ModelCache(directory=tmp_path / name, policy=CachePolicy.READ_WRITE),
        )
        self.answers = answers

    def _ask(self, rows: Any) -> tuple[dict[str, Any], tuple[None, None, None], str]:
        items = [
            {
                "txn_id": r.txn_id,
                "category": self.answers[r.txn_id][0],
                "confidence": self.answers[r.txn_id][1],
                "evidence": r.description[:20],
            }
            for r in rows
            if r.txn_id in self.answers
        ]
        return {"items": items}, (None, None, None), "req_test"


def engine(tmp_path: Path, name: str, answers: dict[str, tuple[str, float]]) -> CategoryClassifier:
    return StubEngine(tmp_path, name, answers)


def test_document_conclusion_wins_over_both_votes(tmp_path: Path) -> None:
    """Вывод аудитора применяется независимо от первоначального отражения."""
    reclassified = ReclassifyAdjustment(
        account_id=ACCOUNT,
        selector=TransactionSelector(txn_id="TXN-P1-0001"),
        new_value="Процентные расходы",
        reason="плата за финансирование",
        source=SOURCE,
    )
    classifier = TransactionClassifier(model=engine(tmp_path, "m", {"TXN-P1-0001": ("opex", 0.99)}))
    result = classifier.run(
        ledger([row("TXN-P1-0001", "Advisory engagement on tariff structuring")], [reclassified]),
        [ACCOUNT],
    )
    record = result.by_id("TXN-P1-0001")
    assert record is not None
    assert record.final_category is C.INTEREST_EXPENSE
    assert record.decision_source is DecisionSource.DOCUMENT
    # Голоса сохраняются: по ним видно, что именно перекрыл документ.
    assert record.model_category is C.OPEX
    assert record.rule_category is C.OTHER


def test_agreement_of_rule_and_model_is_accepted(tmp_path: Path) -> None:
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0002": ("taxes", 0.95)})
    )
    result = classifier.run(
        ledger([row("TXN-P1-0002", "Corporate income tax instalment")]), [ACCOUNT]
    )
    record = result.by_id("TXN-P1-0002")
    assert record is not None
    assert record.decision_source is DecisionSource.AGREEMENT
    assert record.final_category is C.TAXES
    assert not record.is_conflict


def test_disagreement_goes_to_the_verifier(tmp_path: Path) -> None:
    """Уверенное неверное правило — главная опасность, поэтому спор разбирается."""
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0003": ("capex", 0.96)}),
        verifier=engine(tmp_path, "v", {"TXN-P1-0003": ("capex", 0.93)}),
    )
    result = classifier.run(
        ledger([row("TXN-P1-0003", "Quay wall inspection and survey servicing contract")]),
        [ACCOUNT],
    )
    record = result.by_id("TXN-P1-0003")
    assert record is not None
    assert record.is_conflict
    assert record.decision_source is DecisionSource.VERIFIER
    assert record.final_category is C.CAPEX
    assert record.rule_category is C.OPEX
    assert classifier.disputed_rows == ["TXN-P1-0003"]


def test_low_confidence_is_checked_before_it_becomes_unknown(tmp_path: Path) -> None:
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0004": ("taxes", 0.40)}),
        verifier=engine(tmp_path, "v", {"TXN-P1-0004": ("taxes", 0.90)}),
    )
    result = classifier.run(
        ledger([row("TXN-P1-0004", "Corporate income tax instalment")]), [ACCOUNT]
    )
    record = result.by_id("TXN-P1-0004")
    assert record is not None
    assert record.decision_source is DecisionSource.VERIFIER
    assert record.final_category is C.TAXES


def test_nobody_is_sure_leaves_the_row_unknown(tmp_path: Path) -> None:
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0005": ("other", 0.20)}),
        verifier=engine(tmp_path, "v", {"TXN-P1-0005": ("other", 0.30)}),
    )
    result = classifier.run(ledger([row("TXN-P1-0005", "Zhaiyk arrangement")]), [ACCOUNT])
    record = result.by_id("TXN-P1-0005")
    assert record is not None
    assert record.final_category is C.UNKNOWN
    assert record.decision_source is DecisionSource.UNRESOLVED
    assert result.unresolved == (record,)


def test_row_missing_from_the_answer_falls_back_to_the_rule(tmp_path: Path) -> None:
    """Одна испорченная запись не должна стоить разбора остальным пятидесяти."""
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0006": ("taxes", 0.95)}),
    )
    result = classifier.run(
        ledger(
            [
                row("TXN-P1-0006", "Corporate income tax instalment"),
                row("TXN-P1-0007", "Office rent"),
            ]
        ),
        [ACCOUNT],
    )
    assert result.by_id("TXN-P1-0006").decision_source is DecisionSource.AGREEMENT  # type: ignore[union-attr]
    missing = result.by_id("TXN-P1-0007")
    assert missing is not None
    assert missing.decision_source is DecisionSource.RULE_ONLY
    assert missing.final_category is C.RENT
    assert classifier.model.calls[0].missing == ("TXN-P1-0007",)


class DroppingEngine(StubEngine):
    """Первый ответ теряет часть строк, повторный отдаёт всё."""

    def __init__(self, tmp_path: Path, name: str, answers: dict[str, tuple[str, float]]) -> None:
        super().__init__(tmp_path, name, answers)
        self.batches: list[tuple[str, ...]] = []

    def _ask(self, rows: Any) -> tuple[dict[str, Any], tuple[None, None, None], str]:
        self.batches.append(tuple(r.txn_id for r in rows))
        payload, usage, request_id = super()._ask(rows)
        if len(self.batches) == 1:
            payload["items"] = payload["items"][:1]
        return payload, usage, request_id


def test_only_the_missing_rows_are_asked_again(tmp_path: Path) -> None:
    """Повтор один и только по недостающему: терять заёмщика из-за одной строки незачем."""
    answers = {"TXN-P1-0020": ("taxes", 0.95), "TXN-P1-0021": ("rent", 0.95)}
    model = DroppingEngine(tmp_path, "m", answers)
    result = TransactionClassifier(model=model).run(
        ledger(
            [
                row("TXN-P1-0020", "Corporate income tax instalment"),
                row("TXN-P1-0021", "Office rent"),
            ]
        ),
        [ACCOUNT],
    )
    assert model.batches == [("TXN-P1-0020", "TXN-P1-0021"), ("TXN-P1-0021",)]
    assert [c.retry for c in model.calls] == [False, True]
    assert result.by_id("TXN-P1-0021").decision_source is DecisionSource.AGREEMENT  # type: ignore[union-attr]
    assert result.unresolved == ()


def test_retry_is_not_repeated_endlessly(tmp_path: Path) -> None:
    """Молчащая по строке модель не должна вызываться бесконечно."""
    silent = StubEngine(tmp_path, "m", {"TXN-P1-0022": ("taxes", 0.95)})
    result = TransactionClassifier(model=silent).run(
        ledger(
            [
                row("TXN-P1-0022", "Corporate income tax instalment"),
                row("TXN-P1-0023", "Office rent"),
            ]
        ),
        [ACCOUNT],
    )
    assert len(silent.calls) == 2
    assert result.by_id("TXN-P1-0023").decision_source is DecisionSource.RULE_ONLY  # type: ignore[union-attr]


def test_verifier_asks_once_per_borrower(tmp_path: Path) -> None:
    """По строке за раз выходит то же самое, но вдесятеро дольше."""
    rows = [row(f"TXN-P1-00{n}", "Quay wall inspection and survey servicing") for n in (30, 31, 32)]
    answers = {r.txn_id: ("capex", 0.96) for r in rows}
    verifier = StubEngine(tmp_path, "v", answers)
    classifier = TransactionClassifier(model=StubEngine(tmp_path, "m", answers), verifier=verifier)
    result = classifier.run(ledger(rows), [ACCOUNT])

    assert len(verifier.calls) == 1
    assert verifier.calls[0].size == 3
    assert len(classifier.disputed_rows) == 3
    assert all(r.decision_source is DecisionSource.VERIFIER for r in result.records)


def test_document_wording_outside_the_ontology_is_not_silently_dropped(tmp_path: Path) -> None:
    """Вывод аудитора сильнее обоих голосов, и потерять его нельзя."""
    adjustment = ReclassifyAdjustment(
        account_id=ACCOUNT,
        selector=TransactionSelector(txn_id="TXN-P1-0040"),
        new_value="Расходы на озеленение территории",
        reason="вывод аудитора",
        source=SOURCE,
    )
    classifier = TransactionClassifier(model=engine(tmp_path, "m", {"TXN-P1-0040": ("opex", 0.99)}))
    result = classifier.run(ledger([row("TXN-P1-0040", "Office rent")], [adjustment]), [ACCOUNT])
    record = result.by_id("TXN-P1-0040")
    assert record is not None
    assert record.final_category is C.UNKNOWN
    assert record.decision_source is DecisionSource.UNRESOLVED
    assert "не отображается" in record.note
    assert result.unresolved == (record,)


def test_second_run_reads_the_cache(tmp_path: Path) -> None:
    answers = {"TXN-P1-0008": ("taxes", 0.95)}
    rows = [row("TXN-P1-0008", "Corporate income tax instalment")]

    first = engine(tmp_path, "shared", answers)
    TransactionClassifier(model=first).run(ledger(rows), [ACCOUNT])
    assert first.live_calls == 1

    second = engine(tmp_path, "shared", answers)
    TransactionClassifier(model=second).run(ledger(rows), [ACCOUNT])
    assert second.live_calls == 0


def test_order_of_rows_does_not_change_the_batch_key(tmp_path: Path) -> None:
    answers = {"TXN-P1-0009": ("taxes", 0.95), "TXN-P1-0010": ("rent", 0.95)}
    forward = [
        row("TXN-P1-0009", "Corporate income tax instalment"),
        row("TXN-P1-0010", "Office rent"),
    ]

    first = engine(tmp_path, "shared", answers)
    TransactionClassifier(model=first).run(ledger(forward), [ACCOUNT])

    second = engine(tmp_path, "shared", answers)
    TransactionClassifier(model=second).run(ledger(list(reversed(forward))), [ACCOUNT])
    assert second.live_calls == 0


def test_matrix_counts_every_decision(tmp_path: Path) -> None:
    classifier = TransactionClassifier(
        model=engine(tmp_path, "m", {"TXN-P1-0011": ("taxes", 0.95)})
    )
    result = classifier.run(
        ledger([row("TXN-P1-0011", "Corporate income tax instalment")]), [ACCOUNT]
    )
    matrix = result.matrix()
    assert matrix["total"] == 1
    assert matrix["agreement"] == 1
    assert matrix["unknown"] == 0


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        ("Процентные расходы", C.INTEREST_EXPENSE),
        ("Страховые премии", C.INSURANCE_PREMIUM),
        ("Капитальные затраты", C.CAPEX),
        ("Расходы на оплату труда", C.PAYROLL),
    ],
)
def test_document_wording_maps_to_the_ontology(tmp_path: Path, stated: str, expected: C) -> None:
    adjustment = ReclassifyAdjustment(
        account_id=ACCOUNT,
        selector=TransactionSelector(txn_id="TXN-P1-0012"),
        new_value=stated,
        reason="вывод аудитора",
        source=SOURCE,
    )
    classifier = TransactionClassifier(model=engine(tmp_path, "m", {}))
    result = classifier.run(ledger([row("TXN-P1-0012", "Office rent")], [adjustment]), [ACCOUNT])
    record = result.by_id("TXN-P1-0012")
    assert record is not None
    assert record.final_category is expected
