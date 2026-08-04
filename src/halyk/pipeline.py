from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from halyk.models.metrics import RunMetrics
from halyk.run.context import RunContext


@dataclass(frozen=True, slots=True)
class SolveResult:
    submission_path: Path
    metrics: RunMetrics


def solve(context: RunContext, input_path: Path, output_path: Path) -> SolveResult:
    """Полный прогон: архив на входе, Submission.json на выходе.

    Порядок стадий:
      1. инвентаризация архива, хеши, классификация документов
      2. разбор PDF нативно, страницы с непригодным текстовым слоем — в OCR
      3. граф документов: заёмщики, пункты, интервалы действия
      4. компиляция пунктов в Covenant IR и проверка цитат по исходнику
      5. расчёт по реестру транзакций, отбор строк и улики
      6. инварианты, разбор конфликтов верификатором
      7. сборка ответа и проверка по схеме
    """
    raise NotImplementedError(
        "Стадии пайплайна собираются после публикации открытого датасета: "
        "форма реестра и шаблон ответа известны только оттуда."
    )
