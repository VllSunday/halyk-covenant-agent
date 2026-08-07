"""Категория операции для целей ковенантов и то, как она была получена.

Категории плоские и взаимоисключающие: `OPEX` не включает в себя ни оплату труда, ни
аренду, ни налоги. Это не соглашение об удобстве, а прочтение договоров. P1/6.1
сравнивает капитальные затраты с суммой «операционных расходов и арендных платежей» —
если бы аренда входила в операционные расходы, она посчиталась бы дважды. P7/6.1
делит сумму налогов и коммунальных расходов на EBITDA, которая сама определена как
выручка за вычетом операционных расходов, — при вложенности отношение ссылалось бы на
себя. B1/6.2 прямо называет оплату труда и коммунальные услуги отдельными статьями,
которые сравниваются с потолком «по отдельности, а не в совокупности».
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionCategory(StrEnum):
    REVENUE = "revenue"
    OPEX = "opex"
    CAPEX = "capex"
    PAYROLL = "payroll"
    UTILITIES = "utilities"
    RENT = "rent"
    TAXES = "taxes"
    INTEREST_EXPENSE = "interest_expense"
    INSURANCE_PREMIUM = "insurance_premium"
    FINANCING_INFLOW = "financing_inflow"
    OTHER = "other"
    UNKNOWN = "unknown"

    @property
    def is_decided(self) -> bool:
        return self is not TransactionCategory.UNKNOWN


class DecisionSource(StrEnum):
    """Кто решил. Порядок — от самого сильного основания к самому слабому."""

    DOCUMENT = "document"
    AGREEMENT = "agreement"
    RULE_ONLY = "rule_only"
    MODEL_ONLY = "model_only"
    VERIFIER = "verifier"
    UNRESOLVED = "unresolved"


class ClassificationRecord(BaseModel):
    """Решение по одной операции вместе со всеми голосами, которые в него вошли."""

    model_config = ConfigDict(frozen=True)

    txn_id: str
    account_id: str
    final_category: TransactionCategory
    decision_source: DecisionSource

    rule_id: str | None = None
    rule_category: TransactionCategory | None = None
    model_category: TransactionCategory | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None
    verifier_category: TransactionCategory | None = None
    verifier_reason: str | None = None
    note: str = ""

    @property
    def is_conflict(self) -> bool:
        """Правило и модель разошлись. Само по себе не ошибка, но требует разбора."""
        return (
            self.rule_category is not None
            and self.model_category is not None
            and self.rule_category is not self.model_category
        )

    def record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
