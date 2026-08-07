"""Чтение реестра транзакций.

Модуль сознательно не использует pandas: суммы читаются как строки и переводятся в
целые минорные единицы, а float на этом пути запрещён решением 0007.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from halyk.models.transaction import LedgerRow
from halyk.money import Currency, Money

REQUIRED_COLUMNS = ("txn_id", "date", "account_id", "counterparty", "description", "amount")


class LedgerError(ValueError):
    pass


def _amount(raw: str, currency: str) -> Money | None:
    """Пустая сумма — не ошибка: её фактическое значение раскрыто в документах."""
    if not raw.strip():
        return None
    try:
        return Money.from_decimal(raw.strip(), Currency(currency.strip().upper()))
    except ValueError as exc:
        raise LedgerError(f"Не разобрал сумму {raw!r} в валюте {currency!r}") from exc


def read_ledger(path: Path) -> tuple[LedgerRow, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise LedgerError(f"В {path} нет колонок: {', '.join(missing)}")

        rows = []
        for line in reader:
            try:
                parsed = date.fromisoformat(line["date"].strip())
            except ValueError as exc:
                raise LedgerError(f"{line['txn_id']}: дата {line['date']!r} не по ISO") from exc
            rows.append(
                LedgerRow(
                    txn_id=line["txn_id"].strip(),
                    date=parsed,
                    account_id=line["account_id"].strip(),
                    counterparty=line["counterparty"].strip(),
                    description=line["description"].strip(),
                    amount=_amount(line["amount"], line.get("currency", "USD")),
                )
            )
    return tuple(rows)
