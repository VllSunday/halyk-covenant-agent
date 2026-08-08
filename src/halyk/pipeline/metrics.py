"""Сбор метрик прогона по стадиям.

Считается только то, что уже произошло: каждая стадия ведёт собственный журнал
вызовов, и метрики — это его свод, а не второй счётчик рядом с работой. Иначе они
расходятся при первой же правке, и верить перестаёшь обоим.
"""

from __future__ import annotations

from collections.abc import Sequence

from halyk.llm.classify import CategoryClassifier
from halyk.llm.runner import ModelRunner, Role
from halyk.models.metrics import StageMetrics
from halyk.parsing.ocr import OcrCall


def from_runner(runner: ModelRunner, role: Role, unresolved: int = 0) -> StageMetrics:
    """Свод по одной роли структурированного вызова.

    Повтором считается попытка со вторым номером: транспортный повтор ключа не
    меняет и отдельной записью не становится, а смысловой — это уже другой запрос.
    """
    calls = [call for call in runner.calls if call.role is role]
    live = [call for call in calls if not call.cache_hit]
    return StageMetrics(
        calls=len(calls),
        cache_hits=len(calls) - len(live),
        live_calls=len(live),
        tokens_in=sum(call.input_tokens or 0 for call in live),
        tokens_out=sum(call.output_tokens or 0 for call in live),
        seconds=round(sum(call.latency_seconds for call in calls), 3),
        retries=sum(1 for call in calls if call.attempt > 1),
        unresolved=unresolved,
    )


def from_classifier(classifier: CategoryClassifier, unresolved: int = 0) -> StageMetrics:
    usage = classifier.usage()
    return StageMetrics(
        calls=usage["batches"],
        cache_hits=usage["cache_hits"],
        live_calls=usage["live_calls"],
        tokens_in=usage["input_tokens"],
        tokens_out=usage["output_tokens"],
        seconds=usage["seconds_total"],
        retries=sum(1 for call in classifier.calls if call.retry),
        unresolved=unresolved,
    )


def from_ocr(calls: Sequence[OcrCall]) -> StageMetrics:
    live = [call for call in calls if not call.cache_hit]
    return StageMetrics(
        calls=len(calls),
        cache_hits=len(calls) - len(live),
        live_calls=len(live),
        tokens_in=sum(call.input_tokens or 0 for call in live),
        tokens_out=sum(call.output_tokens or 0 for call in live),
        seconds=round(sum(call.latency_seconds for call in calls), 3),
    )
