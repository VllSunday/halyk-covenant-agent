"""Adjustment Ledger: чистая функция от реестра, корректировок и периода."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from halyk.ingest.normalise import CovenantPeriod, normalise
from halyk.models.adjustment import (
    Adjustment,
    AdjustmentStatus,
    ConvertCurrencyAdjustment,
    ExcludeAdjustment,
    ReclassifyAdjustment,
    ReviewNoChangeAdjustment,
    SelectorError,
    SetEffectiveDateAdjustment,
    SetMissingAmountAdjustment,
    TransactionSelector,
)
from halyk.models.document import DocumentKind, DocumentStatus
from halyk.models.source import SourceAuthority, SourceRef
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, CurrencyMismatchError, Money

PERIOD = CovenantPeriod(start=date(2025, 1, 1), end=date(2025, 12, 31))

SOURCE = SourceRef(
    file_hash="a" * 64,
    file_name="notes.pdf",
    page=3,
    kind=DocumentKind.FINANCIAL_NOTES,
    status=DocumentStatus.CURRENT,
    authority=SourceAuthority.AUTHORITATIVE,
    account_id="ACC-7801",
)


def usd(value: str) -> Money:
    return Money.from_decimal(Decimal(value), Currency.USD)


def row(
    txn_id: str,
    amount: Money | None = None,
    when: date = date(2025, 6, 1),
    counterparty: str = "Aktau Holdings LLP",
) -> LedgerRow:
    return LedgerRow(
        txn_id=txn_id,
        date=when,
        account_id="ACC-7801",
        counterparty=counterparty,
        description="Advisory retainer",
        amount=amount,
    )


def make(model: type[Any], **fields: Any) -> Any:
    return model(account_id="ACC-7801", reason="проверка", **{"source": SOURCE, **fields})


def reclassify(txn_id: str = "TXN-P1-0001", new_value: str = "Процентные расходы") -> Adjustment:
    return make(
        ReclassifyAdjustment,
        selector=TransactionSelector(txn_id=txn_id),
        new_value=new_value,
    )


@pytest.fixture
def rows() -> list[LedgerRow]:
    return [
        row("TXN-P1-0001", usd("-100.00")),
        row("TXN-P1-0002", None, counterparty="State Revenue Committee"),
        row("TXN-P1-0003", usd("-592296.10"), counterparty="Irtysh Advisory Bureau"),
    ]


def test_original_rows_are_never_modified(rows: list[LedgerRow]) -> None:
    """Строка выгрузки — свидетельство. Меняется представление, а не она."""
    before = [r.model_copy(deep=True) for r in rows]
    ledger = normalise(rows, [reclassify()], PERIOD)
    assert rows == before
    assert ledger.by_id("TXN-P1-0001").row.amount == usd("-100.00")  # type: ignore[union-attr]


def test_reclassification_sets_the_covenant_category(rows: list[LedgerRow]) -> None:
    ledger = normalise(rows, [reclassify()], PERIOD)
    changed = ledger.by_id("TXN-P1-0001")
    assert changed is not None
    assert changed.covenant_category == "Процентные расходы"
    assert ledger.adjustments[0].status is AdjustmentStatus.APPLIED
    assert changed.adjustments == (ledger.adjustments[0].id,)


def test_missing_amount_is_filled_from_the_document(rows: list[LedgerRow]) -> None:
    ledger = normalise(
        rows,
        [
            make(
                SetMissingAmountAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0002"),
                amount=usd("-486204.19"),
            )
        ],
        PERIOD,
    )
    changed = ledger.by_id("TXN-P1-0002")
    assert changed is not None
    assert changed.amount == usd("-486204.19")
    assert changed.row.amount is None


def test_exclusion_takes_the_transaction_out_of_the_period(rows: list[LedgerRow]) -> None:
    ledger = normalise(
        rows,
        [make(ExcludeAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001"))],
        PERIOD,
    )
    assert ledger.by_id("TXN-P1-0001").in_period is False  # type: ignore[union-attr]


def test_effective_date_outside_the_period_excludes_the_transaction(
    rows: list[LedgerRow],
) -> None:
    """Дата счёта-фактуры не решает: период определяется датой оказания услуг."""
    ledger = normalise(
        rows,
        [
            make(
                SetEffectiveDateAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                effective_date=date(2026, 1, 15),
            )
        ],
        PERIOD,
    )
    changed = ledger.by_id("TXN-P1-0001")
    assert changed is not None
    assert changed.effective_date == date(2026, 1, 15)
    assert changed.in_period is False
    assert changed.row.date == date(2025, 6, 1)


def test_period_comes_from_the_caller(rows: list[LedgerRow]) -> None:
    """Тот же реестр с другим периодом даёт другой ответ, и это должно быть явным."""
    shifted = CovenantPeriod(start=date(2024, 1, 1), end=date(2024, 12, 31))
    ledger = normalise(rows, [], shifted)
    assert all(not txn.in_period for txn in ledger.transactions)
    assert ledger.period.record() == {"start": "2024-01-01", "end": "2024-12-31"}


def test_currency_conversion_uses_the_disclosed_rate() -> None:
    eur = Money.from_decimal(Decimal("-612884.25"), Currency.EUR)
    ledger = normalise(
        [row("TXN-P3-0024", eur, counterparty="Rheinland Katalyse Service GmbH")],
        [
            make(
                ConvertCurrencyAdjustment,
                selector=TransactionSelector(counterparty="Rheinland Katalyse Service GmbH"),
                from_currency=Currency.EUR,
                to_currency=Currency.USD,
                rate=Decimal("1.16"),
            )
        ],
        PERIOD,
    )
    changed = ledger.by_id("TXN-P3-0024")
    assert changed is not None
    assert changed.amount == usd("-710945.73")


def test_selector_by_amount_and_counterparty_ignores_the_sign(rows: list[LedgerRow]) -> None:
    """В документе сумма названа без знака, в реестре знак несёт направление операции."""
    ledger = normalise(
        rows,
        [
            make(
                ReclassifyAdjustment,
                selector=TransactionSelector(
                    counterparty="Irtysh Advisory Bureau", amount=usd("592296.10")
                ),
                new_value="Процентные расходы",
            )
        ],
        PERIOD,
    )
    assert ledger.adjustments[0].txn_id == "TXN-P1-0003"
    assert ledger.by_id("TXN-P1-0003").covenant_category == "Процентные расходы"  # type: ignore[union-attr]


def test_unmatched_adjustment_changes_nothing(rows: list[LedgerRow]) -> None:
    ledger = normalise(rows, [reclassify(txn_id="TXN-P1-9999")], PERIOD)
    assert ledger.adjustments[0].status is AdjustmentStatus.UNMATCHED
    assert not any(txn.is_adjusted for txn in ledger.transactions)
    assert ledger.problems == ledger.adjustments


def test_ambiguous_selector_is_not_applied() -> None:
    """Две подходящие строки — повод остановиться, а не выбрать первую попавшуюся."""
    twins = [row("TXN-P1-0001", usd("-100.00")), row("TXN-P1-0002", usd("-100.00"))]
    ledger = normalise(
        twins,
        [
            make(
                ReclassifyAdjustment,
                selector=TransactionSelector(
                    counterparty="Aktau Holdings LLP", amount=usd("100.00")
                ),
                new_value="Налоги",
            )
        ],
        PERIOD,
    )
    assert ledger.adjustments[0].status is AdjustmentStatus.AMBIGUOUS
    assert ledger.adjustments[0].note == "TXN-P1-0001, TXN-P1-0002"
    assert not any(txn.is_adjusted for txn in ledger.transactions)


def test_missing_amount_does_not_overwrite_a_known_one(rows: list[LedgerRow]) -> None:
    """Строка с суммой — свидетельство. Число из примечаний её не подменяет."""
    ledger = normalise(
        rows,
        [
            make(
                SetMissingAmountAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                amount=usd("-999.00"),
            )
        ],
        PERIOD,
    )
    assert ledger.adjustments[0].status is AdjustmentStatus.INEFFECTIVE
    assert ledger.by_id("TXN-P1-0001").amount == usd("-100.00")  # type: ignore[union-attr]


def test_two_documents_confirming_the_same_change_are_not_a_conflict(
    rows: list[LedgerRow],
) -> None:
    """Совпавшие выводы усиливают друг друга, а не отменяют.

    Обоснование ответа становится сильнее, поэтому второй источник остаётся в
    происхождении операции, хотя число он не меняет.
    """
    first = reclassify(new_value="Налоги")
    second = first.model_copy(update={"source": SOURCE.model_copy(update={"page": 7})})
    ledger = normalise(rows, [first, second], PERIOD)

    assert [a.status for a in ledger.adjustments] == [
        AdjustmentStatus.APPLIED,
        AdjustmentStatus.CORROBORATED,
    ]
    changed = ledger.by_id("TXN-P1-0001")
    assert changed is not None
    assert changed.covenant_category == "Налоги"
    assert changed.adjustments == (first.id, second.id)
    assert ledger.problems == ()


def test_exact_duplicate_is_corroboration_not_a_second_effect(rows: list[LedgerRow]) -> None:
    item = reclassify(new_value="Налоги")
    ledger = normalise(rows, [item, item], PERIOD)
    assert [a.status for a in ledger.adjustments] == [
        AdjustmentStatus.APPLIED,
        AdjustmentStatus.CORROBORATED,
    ]
    assert ledger.by_id("TXN-P1-0001").adjustments == (item.id,)  # type: ignore[union-attr]


def test_exclusion_and_date_transfer_compose_in_any_order(rows: list[LedgerRow]) -> None:
    """Исключённая операция остаётся вне периода, даже если дату перенесли внутрь него."""
    items: list[Adjustment] = [
        make(ExcludeAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001")),
        make(
            SetEffectiveDateAdjustment,
            selector=TransactionSelector(txn_id="TXN-P1-0001"),
            effective_date=date(2025, 3, 1),
        ),
    ]
    forward = normalise(rows, items, PERIOD)
    backward = normalise(rows, list(reversed(items)), PERIOD)
    for ledger in (forward, backward):
        changed = ledger.by_id("TXN-P1-0001")
        assert changed is not None
        assert changed.in_period is False
        assert changed.effective_date == date(2025, 3, 1)


def test_two_documents_arguing_about_one_field_apply_neither(rows: list[LedgerRow]) -> None:
    """Порядок обхода каталога не должен решать, чья версия попадёт в ответ."""
    ledger = normalise(
        rows,
        [reclassify(new_value="Налоги"), reclassify(new_value="Процентные расходы")],
        PERIOD,
    )
    assert {a.status for a in ledger.adjustments} == {AdjustmentStatus.CONFLICTING}
    assert ledger.by_id("TXN-P1-0001").covenant_category is None  # type: ignore[union-attr]
    assert len(ledger.problems) == 2


def test_adjustments_touching_different_fields_do_not_conflict(rows: list[LedgerRow]) -> None:
    ledger = normalise(
        rows,
        [
            reclassify(new_value="Налоги"),
            make(ExcludeAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001")),
        ],
        PERIOD,
    )
    changed = ledger.by_id("TXN-P1-0001")
    assert changed is not None
    assert changed.covenant_category == "Налоги"
    assert changed.in_period is False
    assert len(changed.adjustments) == 2


def test_adjustment_without_effect_is_not_applied() -> None:
    """«Применено» без изменения состояния — самый опасный вид зелёного отчёта.

    Операция за пределами периода и так не участвует в расчёте: исключение её не
    меняет, и отчитываться о применённой корректировке здесь не о чем.
    """
    outside = [row("TXN-P1-0001", usd("-100.00"), when=date(2024, 6, 1))]
    ledger = normalise(
        outside,
        [make(ExcludeAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001"))],
        PERIOD,
    )
    assert ledger.adjustments[0].status is AdjustmentStatus.INEFFECTIVE
    assert ledger.applied == ()
    assert ledger.problems == ledger.adjustments


@pytest.mark.parametrize(
    "status",
    [AdjustmentStatus.REJECTED, AdjustmentStatus.SUPERSEDED, AdjustmentStatus.UNCONFIRMED],
)
def test_non_pending_adjustments_never_reach_the_ledger(
    rows: list[LedgerRow], status: AdjustmentStatus
) -> None:
    ledger = normalise(
        rows,
        [
            make(
                ReclassifyAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                new_value="Налоги",
                status=status,
            )
        ],
        PERIOD,
    )
    assert ledger.adjustments[0].status is status
    assert ledger.by_id("TXN-P1-0001").covenant_category is None  # type: ignore[union-attr]
    assert ledger.problems == ()


def test_review_without_change_never_touches_the_ledger(rows: list[LedgerRow]) -> None:
    ledger = normalise(
        rows,
        [
            make(
                ReviewNoChangeAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                considered="Страховые премии",
            )
        ],
        PERIOD,
    )
    assert ledger.adjustments[0].status is AdjustmentStatus.REJECTED
    assert not any(txn.is_adjusted for txn in ledger.transactions)


def test_result_does_not_depend_on_the_order_of_adjustments(rows: list[LedgerRow]) -> None:
    """Метаморфная проверка: перестановка независимых корректировок ничего не меняет."""
    items: list[Adjustment] = [
        reclassify(new_value="Налоги"),
        make(
            SetMissingAmountAdjustment,
            selector=TransactionSelector(txn_id="TXN-P1-0002"),
            amount=usd("-1.00"),
        ),
    ]
    forward = normalise(rows, items, PERIOD)
    backward = normalise(rows, list(reversed(items)), PERIOD)
    assert {t.txn_id: t.state() for t in forward.transactions} == {
        t.txn_id: t.state() for t in backward.transactions
    }


def test_result_is_reproducible(rows: list[LedgerRow]) -> None:
    items: list[Adjustment] = [reclassify(new_value="Налоги")]
    first = normalise(rows, items, PERIOD)
    second = normalise(rows, items, PERIOD)
    assert [t.record() for t in first.transactions] == [t.record() for t in second.transactions]
    assert [a.record() for a in first.adjustments] == [a.record() for a in second.adjustments]


class TestContract:
    """Корректировку без предмета нельзя собрать — не то что применить."""

    def test_reclassification_requires_a_new_value(self) -> None:
        with pytest.raises(ValidationError):
            make(ReclassifyAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001"))

    def test_reclassification_rejects_an_empty_new_value(self) -> None:
        with pytest.raises(ValidationError):
            make(
                ReclassifyAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                new_value="",
            )

    def test_missing_amount_requires_an_amount(self) -> None:
        with pytest.raises(ValidationError):
            make(SetMissingAmountAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0002"))

    def test_effective_date_requires_a_date(self) -> None:
        with pytest.raises(ValidationError):
            make(SetEffectiveDateAdjustment, selector=TransactionSelector(txn_id="TXN-P1-0001"))

    @pytest.mark.parametrize("rate", [Decimal(0), Decimal("-1.16")])
    def test_conversion_requires_a_positive_rate(self, rate: Decimal) -> None:
        with pytest.raises(ValidationError):
            make(
                ConvertCurrencyAdjustment,
                selector=TransactionSelector(counterparty="X"),
                from_currency=Currency.EUR,
                to_currency=Currency.USD,
                rate=rate,
            )

    def test_conversion_requires_both_currencies(self) -> None:
        with pytest.raises(ValidationError):
            make(
                ConvertCurrencyAdjustment,
                selector=TransactionSelector(counterparty="X"),
                rate=Decimal("1.16"),
            )

    def test_conversion_rejects_the_same_currency_on_both_sides(self) -> None:
        """Пересчёт доллара в доллар по курсу 2 — это умножение суммы, а не пересчёт."""
        with pytest.raises((CurrencyMismatchError, ValidationError)):
            make(
                ConvertCurrencyAdjustment,
                selector=TransactionSelector(counterparty="X"),
                from_currency=Currency.USD,
                to_currency=Currency.USD,
                rate=Decimal(2),
            )

    def test_review_without_change_cannot_be_applied(self) -> None:
        with pytest.raises(ValidationError):
            make(
                ReviewNoChangeAdjustment,
                selector=TransactionSelector(txn_id="TXN-P1-0001"),
                status=AdjustmentStatus.APPLIED,
            )

    @pytest.mark.parametrize(
        "status",
        [AdjustmentStatus.REJECTED, AdjustmentStatus.SUPERSEDED, AdjustmentStatus.UNCONFIRMED],
    )
    def test_review_keeps_the_reason_it_was_not_applied(self, status: AdjustmentStatus) -> None:
        item = make(
            ReviewNoChangeAdjustment,
            selector=TransactionSelector(txn_id="TXN-P1-0001"),
            status=status,
        )
        assert item.status is status

    def test_empty_selector_is_rejected(self) -> None:
        with pytest.raises((SelectorError, ValidationError)):
            TransactionSelector()


class TestIdentity:
    """Идентификатор обязан различать корректировки, дающие разный результат."""

    def convert(self, rate: str) -> Adjustment:
        return make(
            ConvertCurrencyAdjustment,
            selector=TransactionSelector(counterparty="Rheinland Katalyse Service GmbH"),
            from_currency=Currency.EUR,
            to_currency=Currency.USD,
            rate=Decimal(rate),
        )

    def test_different_rates_are_different_adjustments(self) -> None:
        assert self.convert("1.1").id != self.convert("1.2").id

    def test_same_rate_gives_the_same_identifier(self) -> None:
        assert self.convert("1.16").id == self.convert("1.16").id

    def test_different_amounts_are_different_adjustments(self) -> None:
        first = make(
            SetMissingAmountAdjustment,
            selector=TransactionSelector(txn_id="TXN-P1-0002"),
            amount=usd("-1.00"),
        )
        second = first.model_copy(update={"amount": usd("-2.00")})
        assert first.id != second.id

    def test_different_dates_are_different_adjustments(self) -> None:
        first = make(
            SetEffectiveDateAdjustment,
            selector=TransactionSelector(txn_id="TXN-P1-0001"),
            effective_date=date(2026, 1, 15),
        )
        second = first.model_copy(update={"effective_date": date(2026, 3, 20)})
        assert first.id != second.id

    def test_different_previous_values_are_different_adjustments(self) -> None:
        first = reclassify()
        second = first.model_copy(update={"old_value": "Консультационные услуги"})
        assert first.id != second.id

    def test_status_does_not_change_the_identifier(self) -> None:
        """Иначе одна и та же корректировка меняла бы имя по ходу прогона."""
        adjustment = reclassify()
        assert adjustment.id == adjustment.resolved(AdjustmentStatus.APPLIED, "TXN-P1-0001").id
