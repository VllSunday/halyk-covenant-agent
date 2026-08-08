"""Светофор по ячейке: на что смотреть человеку в первую очередь.

Это оценка риска, а не точности. Точность известна только против ключа, а ключа у
приватного прогона нет и не будет; называть одно другим — самый дорогой способ себя
обмануть, потому что зелёная ячейка перестанет означать «проверено» и начнёт означать
«верно».

Красный — ячейки не будет в ответе. Жёлтый — ответ есть, но что-то в нём просит
взгляда: нарушение без улики, низкая уверенность компиляции, пустая выборка строк.
Зелёный — ничего из перечисленного, и не более того.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from halyk.audit.view import CellView, RunView

# Ниже этого значения формулировка пункта допускала второе прочтение — так сказала
# сама модель. Число не калибровано и служит поводом посмотреть, а не вердиктом.
CONFIDENCE_FLOOR = 0.75

# Диагностика исполнителя, которую стоит поднять в аудит: пустая выборка бывает
# законной («ноль платежей связанной стороне»), а бывает промахом классификации.
_EMPTY_SELECTION = "empty_selection"


class Grade(StrEnum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True, slots=True)
class Graded:
    """Ячейка вместе с причинами, по которым ей выставлен цвет."""

    name: str
    grade: Grade
    reasons: tuple[str, ...]
    cell: CellView | None = None


def _cell_reasons(cell: CellView) -> list[str]:
    reasons: list[str] = []
    if cell.failed_invariants:
        reasons.append("инварианты: " + ", ".join(cell.failed_invariants))
    if cell.status == "BREACH" and cell.evidence_txn_id is None:
        reasons.append("нарушение без выделенной улики")
    if cell.confidence < CONFIDENCE_FLOOR:
        reasons.append(f"уверенность компиляции {cell.confidence:.2f}")
    if not cell.sources:
        reasons.append("у ответа нет адреса источника")
    reasons.extend(f"диагностика: {note}" for note in cell.diagnostics if _EMPTY_SELECTION in note)
    return reasons


def grade_cell(cell: CellView) -> Graded:
    reasons = _cell_reasons(cell)
    if cell.failed_invariants or not cell.sources:
        return Graded(name=cell.name, grade=Grade.RED, reasons=tuple(reasons), cell=cell)
    if reasons:
        return Graded(name=cell.name, grade=Grade.YELLOW, reasons=tuple(reasons), cell=cell)
    return Graded(name=cell.name, grade=Grade.GREEN, reasons=(), cell=cell)


def grade_run(view: RunView) -> list[Graded]:
    """Все ячейки прогона: посчитанные и те, которых в ответе нет.

    Непосчитанная ячейка тоже получает строку — иначе таблица из одних зелёных
    выглядела бы как полный ответ.
    """
    graded = [grade_cell(cell) for _, cell in sorted(view.cells.items())]
    for name in view.missing_cells:
        graded.append(
            Graded(
                name=name,
                grade=Grade.RED,
                reasons=tuple(_explanations(view, name)) or ("ячейки нет в ответе",),
            )
        )
    return sorted(graded, key=lambda item: item.name)


def _explanations(view: RunView, name: str) -> list[str]:
    """Причины отказа по ячейке.

    Проблема адресуется либо самой ячейкой, либо её требованием (`E1/6.2/req-1`),
    либо целым заёмщиком, если у того не собрался ни один пункт.
    """
    scenario = name.split("/", 1)[0]
    return [
        f"{problem['code']}: {problem['detail']}"
        for problem in view.problems
        if problem["subject"] in (name, scenario) or problem["subject"].startswith(f"{name}/")
    ]


def counts(graded: list[Graded]) -> dict[str, int]:
    return {value.value: sum(1 for item in graded if item.grade is value) for value in Grade}
