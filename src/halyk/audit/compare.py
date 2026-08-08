"""Сравнение двух прогонов.

Нужно ровно для одного: понять, что сделало изменение. На приватных данных ключа нет,
и единственный доступный вопрос — «что поменялось по сравнению с прошлым разом и не
стало ли хуже». Ответ на него не заменяет точность и не притворяется ею.

Регрессией считается потеря: пропавшая ячейка, ухудшившийся цвет, новое открытое
требование. Изменившееся число регрессией само по себе не является — ради него правки
и делаются, — но в отчёт попадает всегда.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halyk.audit.grade import Grade, Graded, grade_run
from halyk.audit.view import CellView, RunView

_ORDER = {Grade.GREEN: 0, Grade.YELLOW: 1, Grade.RED: 2}


@dataclass(frozen=True, slots=True)
class CellDiff:
    """Что изменилось в одной ячейке."""

    name: str
    changes: tuple[str, ...]
    baseline_grade: Grade
    candidate_grade: Grade

    @property
    def is_regression(self) -> bool:
        return _ORDER[self.candidate_grade] > _ORDER[self.baseline_grade]


@dataclass(frozen=True, slots=True)
class Comparison:
    """Итог сравнения: изменения по ячейкам и общий вывод о регрессии."""

    diffs: tuple[CellDiff, ...] = ()
    lost: tuple[str, ...] = ()
    gained: tuple[str, ...] = ()
    new_open_requirements: tuple[str, ...] = ()
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def regressions(self) -> tuple[CellDiff, ...]:
        return tuple(diff for diff in self.diffs if diff.is_regression)

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions or self.lost or self.new_open_requirements)

    @property
    def changed(self) -> tuple[CellDiff, ...]:
        return tuple(diff for diff in self.diffs if diff.changes)


def _field_changes(before: CellView, after: CellView) -> list[str]:
    compared = (
        ("вердикт", before.status, after.status),
        ("число", str(before.actual), str(after.actual)),
        ("улика", before.evidence_txn_id or "—", after.evidence_txn_id or "—"),
        ("порог", str(before.threshold), str(after.threshold)),
        ("дерево", before.ir_hash[:12], after.ir_hash[:12]),
        ("источники", ", ".join(before.sources), ", ".join(after.sources)),
        ("строк расчёта", str(len(before.contributing_rows)), str(len(after.contributing_rows))),
    )
    return [f"{name}: {old} → {new}" for name, old, new in compared if old != new]


def _open_requirements(view: RunView) -> set[str]:
    return {item["subject"] for item in view.problems_with("requirement_is_")}


def compare(baseline: RunView, candidate: RunView) -> Comparison:
    """Сравнить прогоны по ячейкам, цветам и оставшимся требованиям."""
    before = {item.name: item for item in grade_run(baseline)}
    after = {item.name: item for item in grade_run(candidate)}

    diffs = [
        CellDiff(
            name=name,
            changes=tuple(_changes(baseline, candidate, name)),
            baseline_grade=before[name].grade,
            candidate_grade=after[name].grade,
        )
        for name in sorted(set(before) & set(after))
    ]
    counts = {
        grade.value: (
            sum(1 for item in before.values() if item.grade is grade),
            sum(1 for item in after.values() if item.grade is grade),
        )
        for grade in Grade
    }
    return Comparison(
        diffs=tuple(diffs),
        lost=tuple(sorted(_names(baseline) - _names(candidate))),
        gained=tuple(sorted(_names(candidate) - _names(baseline))),
        new_open_requirements=tuple(
            sorted(_open_requirements(candidate) - _open_requirements(baseline))
        ),
        counts=counts,
    )


def _names(view: RunView) -> set[str]:
    return {f"{scenario}/{clause}" for scenario, clause in view.cells}


def _changes(baseline: RunView, candidate: RunView, name: str) -> list[str]:
    scenario, clause = name.split("/", 1)
    before = baseline.cells.get((scenario, clause))
    after = candidate.cells.get((scenario, clause))
    if before is None or after is None:
        return []
    return _field_changes(before, after)


def graded_pairs(baseline: RunView, candidate: RunView) -> list[tuple[Graded, Graded]]:
    """Пары оценок по общим ячейкам — для таблицы в терминале."""
    before = {item.name: item for item in grade_run(baseline)}
    after = {item.name: item for item in grade_run(candidate)}
    return [(before[name], after[name]) for name in sorted(set(before) & set(after))]
