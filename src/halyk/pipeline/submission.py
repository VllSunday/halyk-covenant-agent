"""Сборка ответа из посчитанных ячеек.

Два отказа, ради которых этот модуль отдельный.

Состав ячеек берётся из шаблона и сверяется с ним же. Пропущенная ячейка стоит
столько же, сколько неверная, а лишняя не засчитывается вовсе, и заметить это надо
здесь, а не по итогам проверки уже отправленного файла.

Незаполненная ячейка не сериализуется. `null` в `status` или `actual` — это ноль
баллов при внешне полном файле, поэтому попытка собрать такую ячейку прекращает
сборку: неполный ответ должен выглядеть как ошибка прогона, а не как ответ.
"""

from __future__ import annotations

from collections.abc import Mapping

from halyk.config import Settings
from halyk.execution.executor import Outcome
from halyk.models.result import Verdict
from halyk.models.submission import CovenantCell, Submission
from halyk.output.template import SubmissionTemplate


class IncompleteCellError(ValueError):
    """Ячейка без вердикта или без числа. Такое в ответ не уходит."""


class CompositionError(ValueError):
    """Состав ячеек разошёлся с шаблоном организаторов."""


def to_cell(outcome: Outcome) -> CovenantCell:
    if outcome.status is None or outcome.actual is None:
        raise IncompleteCellError(
            f"{outcome.scenario_id}/{outcome.clause_id}: вердикт {outcome.status}, "
            f"значение {outcome.actual}"
        )
    return CovenantCell(
        status=Verdict(outcome.status),
        actual=outcome.actual,
        evidence_txn_id=outcome.evidence_txn_id,
    )


def build_submission(
    template: SubmissionTemplate,
    settings: Settings,
    answers: Mapping[tuple[str, str], Outcome],
) -> Submission:
    """Ответ по шаблону. Порядок ячеек — шаблонный, чтобы файл был воспроизводим."""
    expected = template.key_set()
    missing = sorted(expected - set(answers))
    extra = sorted(set(answers) - expected)
    if missing or extra:
        raise CompositionError(
            "состав ячеек не совпал с шаблоном: "
            f"нет {[f'{s}/{c}' for s, c in missing]}, лишние {[f'{s}/{c}' for s, c in extra]}"
        )

    grouped: dict[str, dict[str, CovenantCell]] = {}
    for scenario, clause in template.cells:
        grouped.setdefault(scenario, {})[clause] = to_cell(answers[scenario, clause])

    return Submission(
        team=settings.team,
        contact_email=settings.contact_email,
        model=settings.compiler.name,
        answers=grouped,
    )
