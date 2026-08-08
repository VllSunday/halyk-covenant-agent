"""Что произошло за прогон. Отдельно от манифеста: тот описывает вход, этот — работу."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class StageMetrics(BaseModel):
    """Расход одной роли за прогон.

    Роли считаются порознь, а не одной суммой: распознавание, классификация,
    компиляция и дочитывание отличаются и ценой вызова, и тем, что означает их
    повтор. Слитые в общий счётчик, они не отвечают на единственный вопрос, ради
    которого метрики и собираются, — где именно ушли деньги и время.
    """

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    cache_hits: int = 0
    live_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    # Повтор здесь смысловой: тот же вопрос, заданный заново после негодного ответа.
    retries: int = 0
    unresolved: int = 0


class RunMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    stages: dict[str, StageMetrics] = Field(default_factory=dict)

    documents_total: int = 0
    pages_total: int = 0
    ocr_pages: int = 0
    transactions_total: int = 0

    borrowers_total: int = 0
    covenants_found: int = 0
    covenants_answered: int = 0

    invariants_passed: int = 0
    invariants_failed: int = 0
    verifier_calls: int = 0
    low_confidence_evidence: int = 0

    llm_calls: int = 0
    cache_hits: int = 0
    live_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: Decimal = Decimal(0)

    runtime_seconds: float = 0.0

    def coverage(self) -> float:
        """Доля ковенантов с ответом. Пропущенный ковенант дороже неверного."""
        if self.covenants_found == 0:
            return 0.0
        return self.covenants_answered / self.covenants_found
