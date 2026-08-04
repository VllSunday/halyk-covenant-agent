"""Covenant IR — то, что модель выдаёт вместо ответа. Обоснование в docs/adr/0001."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from halyk.models.source import SourceRef
from halyk.money import Rounding


class Aggregation(StrEnum):
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    AVERAGE = "average"
    RATIO = "ratio"
    BALANCE_END = "balance_end"
    BALANCE_AVERAGE = "balance_average"
    DAYS_BETWEEN = "days_between"


class Operator(StrEnum):
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    EQ = "eq"
    NE = "ne"

    def holds(self, measured: Decimal, threshold: Decimal) -> bool:
        match self:
            case Operator.GE:
                return measured >= threshold
            case Operator.GT:
                return measured > threshold
            case Operator.LE:
                return measured <= threshold
            case Operator.LT:
                return measured < threshold
            case Operator.EQ:
                return measured == threshold
            case Operator.NE:
                return measured != threshold


class Unit(StrEnum):
    MONEY = "money"
    RATIO = "ratio"
    PERCENT = "percent"
    COUNT = "count"
    DAYS = "days"


class EvidenceRule(StrEnum):
    """Как код выбирает улику из строк расчёта. Обоснование в docs/adr/0005."""

    FIRST_VIOLATION = "first_violation"
    LARGEST_CONTRIBUTOR = "largest_contributor"
    NOT_APPLICABLE = "not_applicable"


class Period(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: date
    end: date
    start_inclusive: bool = True
    end_inclusive: bool = True

    @model_validator(mode="after")
    def check_order(self) -> Period:
        if self.end < self.start:
            raise ValueError(f"Период кончается раньше, чем начинается: {self.start}..{self.end}")
        return self


class Scope(BaseModel):
    """Что попадает в расчёт до применения фильтров."""

    model_config = ConfigDict(frozen=True)

    accounts: tuple[str, ...] = ()
    transaction_types: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    counterparties: tuple[str, ...] = ()


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    operator: Operator
    value: str | int | Decimal
    negate: bool = False


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    operator: Operator
    threshold: Decimal
    unit: Unit
    currency: str | None = None

    @model_validator(mode="after")
    def check_currency(self) -> Comparison:
        if self.unit is Unit.MONEY and not self.currency:
            raise ValueError("Денежный порог без валюты сравнивать нельзя")
        return self


class RoundingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Rounding = Rounding.HALF_UP
    scale: int = Field(default=2, ge=0, le=10)


class CovenantIR(BaseModel):
    """Исполнимая спецификация одного ковенанта.

    Всё, что нужно для расчёта, лежит здесь; исполнителю не требуется ни исходный
    текст, ни обращение к модели.
    """

    model_config = ConfigDict(frozen=True)

    borrower_id: str
    covenant_id: str
    clause_id: str
    metric: str

    scope: Scope = Scope()
    filters: tuple[Filter, ...] = ()
    exclusions: tuple[Filter, ...] = ()
    measurement_period: Period
    aggregation: Aggregation
    comparison: Comparison
    rounding: RoundingSpec = RoundingSpec()
    evidence_rule: EvidenceRule

    effective_from: date | None = None
    effective_to: date | None = None

    source_refs: tuple[SourceRef, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_evidence_rule(self) -> CovenantIR:
        aggregate = {Aggregation.AVERAGE, Aggregation.BALANCE_AVERAGE, Aggregation.RATIO}
        if self.aggregation in aggregate and self.evidence_rule is EvidenceRule.FIRST_VIOLATION:
            raise ValueError(
                f"Для агрегата {self.aggregation} нет отдельной нарушающей строки, "
                f"правило выбора улики должно быть другим"
            )
        return self
