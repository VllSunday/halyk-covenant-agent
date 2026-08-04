"""Аудиторский след. Одна запись на ковенант, читается отчётом и командой explain."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from halyk.models.result import CalculationStep, Verdict
from halyk.models.source import SourceRef


class InvariantCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str | None = None


class LineageRecord(BaseModel):
    """Полная цепочка от текста договора до ответа.

    Это единственный источник правды для отчёта, команды explain и проверки
    инвариантов: три потребителя читают один артефакт, а не считают заново.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    recorded_at: datetime

    borrower_id: str
    covenant_id: str
    clause_id: str
    clause_effective_from: str | None = None

    source_refs: tuple[SourceRef, ...]
    ir_hash: str
    ir_confidence: float = Field(ge=0.0, le=1.0)

    calculator: str
    steps: tuple[CalculationStep, ...] = ()
    contributing_row_ids: tuple[str, ...] = ()
    violating_row_ids: tuple[str, ...] = ()

    measured_value: Decimal
    threshold: Decimal
    operator: str
    verdict: Verdict
    evidence_transaction_id: str | None = None

    invariants: tuple[InvariantCheck, ...] = ()
    verifier_used: bool = False
    notes: tuple[str, ...] = ()

    def failed_invariants(self) -> tuple[InvariantCheck, ...]:
        return tuple(check for check in self.invariants if not check.passed)
