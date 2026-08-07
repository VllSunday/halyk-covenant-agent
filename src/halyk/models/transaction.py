"""Строка реестра. Разбирается один раз и дальше живёт в этом виде."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from halyk.money import Money

# `TXN-<scenario>-<номер>`: без всех трёх частей префикс сценария не определён.
_TXN_ID_PARTS = 3


class LedgerRow(BaseModel):
    """Одна операция реестра.

    `amount` необязателен: в выгрузке встречаются строки с пустой суммой, и это не
    порча данных, а часть задачи — фактическая сумма раскрыта в другом документе.
    Подставлять сюда ноль нельзя, иначе разница между «нет данных» и «нулевой
    оборот» потеряется.
    """

    model_config = ConfigDict(frozen=True)

    txn_id: str
    date: date
    account_id: str
    counterparty: str
    description: str
    amount: Money | None = None

    @property
    def scenario_id(self) -> str:
        """Префикс идентификатора операции. Правило зафиксировано в условии задачи."""
        parts = self.txn_id.split("-")
        return parts[1] if len(parts) >= _TXN_ID_PARTS else ""
