"""Применение корректировок к реестру.

Функция чистая: одни и те же строки, корректировки и период дают один и тот же
результат, сколько бы раз его ни пересчитали. Исходные строки не меняются —
нормализованная операция носит их с собой, поэтому в объяснении ответа всегда можно
показать, что пришло из выгрузки, а что из документа.

Применённой считается только корректировка, что-то изменившая. Документ, чей вывод
не отразился на операции, — это не «применено», а повод разобраться: либо мы не так
его прочитали, либо не ту строку нашли.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from halyk.knowledge.kyc import normalise_counterparty
from halyk.models.adjustment import (
    Adjustment,
    AdjustmentStatus,
    ConvertCurrencyAdjustment,
    ExcludeAdjustment,
    NormalisedTransaction,
    ReclassifyAdjustment,
    SetEffectiveDateAdjustment,
    SetMissingAmountAdjustment,
)
from halyk.models.transaction import LedgerRow
from halyk.money import convert


@dataclass(frozen=True, slots=True)
class CovenantPeriod:
    """Период, за который проверяются ковенанты.

    Значение приходит из договора и записывается в артефакт прогона: без него
    неясно, почему операция ноябрьского аванса оказалась вне периода.
    """

    start: date
    end: date

    def contains(self, when: date) -> bool:
        return self.start <= when <= self.end

    def record(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class NormalisedLedger:
    """Реестр после корректировок вместе с судьбой каждой из них."""

    transactions: tuple[NormalisedTransaction, ...]
    adjustments: tuple[Adjustment, ...]
    period: CovenantPeriod

    @property
    def applied(self) -> tuple[Adjustment, ...]:
        return tuple(a for a in self.adjustments if a.status is AdjustmentStatus.APPLIED)

    @property
    def problems(self) -> tuple[Adjustment, ...]:
        """Корректировки, которые не дошли до реестра не по решению документа."""
        return tuple(a for a in self.adjustments if a.status.is_problem)

    def by_id(self, txn_id: str) -> NormalisedTransaction | None:
        return next((t for t in self.transactions if t.txn_id == txn_id), None)


def _matches(row: LedgerRow, adjustment: Adjustment) -> bool:
    selector = adjustment.selector
    if row.account_id != adjustment.account_id:
        return False
    if selector.txn_id is not None:
        return row.txn_id == selector.txn_id
    if selector.counterparty is not None and normalise_counterparty(
        row.counterparty
    ) != normalise_counterparty(selector.counterparty):
        return False
    # Знак в реестре несёт направление операции, а в документе сумма названа без
    # него: «выплачено $592,296.10» — это -592296.10 в выгрузке.
    if selector.amount is not None and (
        row.amount is None or abs(row.amount) != abs(selector.amount)
    ):
        return False
    if isinstance(adjustment, ConvertCurrencyAdjustment):
        return row.amount is not None and row.amount.currency is adjustment.from_currency
    return True


def _apply(
    transaction: NormalisedTransaction, adjustment: Adjustment, period: CovenantPeriod
) -> NormalisedTransaction:
    update: dict[str, object] = {"adjustments": (*transaction.adjustments, adjustment.id)}
    match adjustment:
        case ReclassifyAdjustment():
            update["covenant_category"] = adjustment.new_value
        case SetMissingAmountAdjustment():
            # Заполняем только пустое место. Строка с суммой — свидетельство, и
            # подменять её числом из примечаний нельзя: расхождение разбирают.
            if transaction.amount is None:
                update["amount"] = adjustment.amount
        case ExcludeAdjustment():
            update["excluded"] = True
            update["in_period"] = False
        case SetEffectiveDateAdjustment():
            # `excluded` держится отдельно, чтобы порядок исключения и переноса даты
            # не решал исход: исключённая операция остаётся вне периода в любом.
            update["effective_date"] = adjustment.effective_date
            update["in_period"] = not transaction.excluded and period.contains(
                adjustment.effective_date
            )
        case ConvertCurrencyAdjustment():
            if transaction.amount is not None:
                update["amount"] = convert(
                    transaction.amount, adjustment.rate, adjustment.to_currency
                )
    return transaction.model_copy(update=update)


def _corroborate(transaction: NormalisedTransaction, adjustment_id: str) -> NormalisedTransaction:
    """Записать второй источник того же вывода, не трогая значения."""
    if adjustment_id in transaction.adjustments:
        return transaction
    return transaction.model_copy(update={"adjustments": (*transaction.adjustments, adjustment_id)})


def _claims(
    candidates: Sequence[tuple[Adjustment, LedgerRow]],
) -> defaultdict[tuple[str, str], set[str]]:
    """Какие значения каждого поля операции запрашивают документы."""
    claims: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for adjustment, row in candidates:
        for field, value in adjustment.effect().items():
            claims[(row.txn_id, field)].add(json.dumps(value, sort_keys=True))
    return claims


def _conflicts(candidates: Sequence[tuple[Adjustment, LedgerRow]]) -> frozenset[str]:
    """Корректировки, спорящие с другой о значении того же поля той же операции.

    Спор — это разные значения. Два документа, называющие одно и то же, друг друга
    подтверждают, и объявлять их конфликтом значит потерять оба вывода. А вот
    молчаливое «побеждает последняя» при расхождении недопустимо: порядок файлов в
    каталоге не должен решать, какая версия суммы попадёт в ответ.
    """
    claims = _claims(candidates)
    return frozenset(
        adjustment.id
        for adjustment, row in candidates
        if any(len(claims[(row.txn_id, field)]) > 1 for field in adjustment.effect())
    )


def normalise(
    rows: Sequence[LedgerRow],
    adjustments: Sequence[Adjustment],
    period: CovenantPeriod,
) -> NormalisedLedger:
    """Собрать нормализованный реестр и проставить каждой корректировке итог."""
    transactions = {
        row.txn_id: NormalisedTransaction(
            row=row,
            amount=row.amount,
            effective_date=row.date,
            in_period=period.contains(row.date),
        )
        for row in rows
    }

    # Сначала сопоставление, потом применение: спор двух документов об одной строке
    # виден только когда известны все претенденты.
    matched: list[tuple[Adjustment, LedgerRow]] = []
    resolved: list[Adjustment] = []
    for adjustment in adjustments:
        if adjustment.status is not AdjustmentStatus.PENDING:
            # Отклонённое, заменённое и неподтверждённое до реестра не доходит, но
            # остаётся в отчёте: «рассмотрели и отказали» — тоже результат.
            resolved.append(adjustment)
            continue

        candidates = [row for row in rows if _matches(row, adjustment)]
        if not candidates:
            resolved.append(adjustment.resolved(AdjustmentStatus.UNMATCHED))
        elif len(candidates) > 1:
            resolved.append(
                adjustment.resolved(
                    AdjustmentStatus.AMBIGUOUS,
                    note=", ".join(row.txn_id for row in candidates),
                )
            )
        else:
            matched.append((adjustment, candidates[0]))

    conflicting = _conflicts(matched)
    already: dict[tuple[str, str, str], str] = {}
    for adjustment, row in matched:
        if adjustment.id in conflicting:
            resolved.append(
                adjustment.resolved(
                    AdjustmentStatus.CONFLICTING,
                    txn_id=row.txn_id,
                    note="другой документ требует по этому полю иного значения",
                )
            )
            continue

        keys = [
            (row.txn_id, field, json.dumps(value, sort_keys=True))
            for field, value in adjustment.effect().items()
        ]
        if keys and all(key in already for key in keys):
            # Тот же вывод из второго источника. Он ничего не добавляет к числу, но
            # добавляет к его обоснованию, поэтому остаётся в происхождении операции.
            first = already[keys[0]]
            transactions[row.txn_id] = _corroborate(transactions[row.txn_id], adjustment.id)
            resolved.append(
                adjustment.resolved(
                    AdjustmentStatus.CORROBORATED, txn_id=row.txn_id, note=f"подтверждает {first}"
                )
            )
            continue

        before = transactions[row.txn_id]
        after = _apply(before, adjustment, period)
        if after.state() == before.state():
            resolved.append(
                adjustment.resolved(
                    AdjustmentStatus.INEFFECTIVE,
                    txn_id=row.txn_id,
                    note="состояние операции не изменилось",
                )
            )
            continue

        transactions[row.txn_id] = after
        already.update(dict.fromkeys(keys, adjustment.id))
        resolved.append(adjustment.resolved(AdjustmentStatus.APPLIED, txn_id=row.txn_id))

    return NormalisedLedger(
        transactions=tuple(transactions.values()),
        adjustments=tuple(resolved),
        period=period,
    )
