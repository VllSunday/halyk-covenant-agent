from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Verdict(StrEnum):
    COMPLIANT = "compliant"
    VIOLATED = "violated"


class CalculationStep(BaseModel):
    """Шаг расчёта в человекочитаемом виде — попадает в карточку решения."""

    model_config = ConfigDict(frozen=True)

    description: str
    rows_before: int
    rows_after: int


class CalculationResult(BaseModel):
    """Результат калькулятора вместе с построчной привязкой.

    Улика выбирается только из этих строк, поэтому число и транзакция не могут
    разойтись. Обоснование в docs/adr/0005.
    """

    model_config = ConfigDict(frozen=True)

    measured_value: Decimal
    contributing_row_ids: tuple[str, ...]
    violating_row_ids: tuple[str, ...]
    steps: tuple[CalculationStep, ...] = ()

    def has_rows(self) -> bool:
        return bool(self.contributing_row_ids)


class CovenantAnswer(BaseModel):
    """Три компонента ответа по одному ковенанту.

    Пустая улика — допустимое значение: выдумывать транзакцию ради заполнения поля
    нельзя, уверенные вердикт и число при этом сохраняются.
    """

    model_config = ConfigDict(frozen=True)

    borrower_id: str
    covenant_id: str
    verdict: Verdict
    value: Decimal
    evidence_transaction_id: str | None = None
    evidence_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
