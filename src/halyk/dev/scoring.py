"""Скоринг ответа по правилам организаторов. Только для разработки.

Формула взята из раздела «Оценка» в CASE: 0.50 за вердикт, 0.30 за число по
убывающей шкале и 0.20 за улику. Веса ячеек по сложности организаторы не
раскрывают, поэтому итог здесь невзвешенный — он годится, чтобы сравнивать версии
между собой, но не совпадёт с публичным рейтингом.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from halyk.models.submission import Submission

STATUS_WEIGHT = Decimal("0.50")
ACTUAL_WEIGHT = Decimal("0.30")
EVIDENCE_WEIGHT = Decimal("0.20")
TOLERANCE = Decimal("0.05")

CellKey = tuple[str, str]


class GroundTruthError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExpectedCell:
    status: str
    actual: Decimal
    evidence_txn_id: str | None


@dataclass(frozen=True, slots=True)
class CellScore:
    scenario: str
    covenant: str
    expected: ExpectedCell
    status: str | None
    actual: Decimal | None
    evidence_txn_id: str | None
    status_points: Decimal
    actual_points: Decimal
    evidence_points: Decimal
    note: str

    @property
    def total(self) -> Decimal:
        return self.status_points + self.actual_points + self.evidence_points

    @property
    def relative_error(self) -> Decimal | None:
        if self.actual is None or self.expected.actual == 0:
            return None
        return abs(self.actual - self.expected.actual) / abs(self.expected.actual)


@dataclass(frozen=True, slots=True)
class ScoreReport:
    cells: tuple[CellScore, ...]
    missing: tuple[CellKey, ...] = ()
    unexpected: tuple[CellKey, ...] = ()

    @property
    def is_comparable(self) -> bool:
        """Совпадает ли набор ячеек с ключом.

        Пропущенные ячейки видны и так — они дают ноль. А вот лишние иначе прошли бы
        незамеченными: итерация идёт по ключу, и ответ с посторонней ячейкой набрал
        бы полный балл. Для скорера, по которому мы принимаем решения, это худший
        вид ошибки — ложное подтверждение.
        """
        return not self.missing and not self.unexpected

    @property
    def total(self) -> Decimal:
        return sum((cell.total for cell in self.cells), Decimal(0))

    @property
    def max_total(self) -> Decimal:
        return Decimal(len(self.cells))

    @property
    def components(self) -> dict[str, Decimal]:
        return {
            "status": sum((c.status_points for c in self.cells), Decimal(0)),
            "actual": sum((c.actual_points for c in self.cells), Decimal(0)),
            "evidence": sum((c.evidence_points for c in self.cells), Decimal(0)),
        }

    @property
    def exact_cells(self) -> int:
        return sum(1 for cell in self.cells if cell.total == Decimal(1))


def load_ground_truth(path: Path) -> dict[CellKey, ExpectedCell]:
    document: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("scenarios"), dict):
        raise GroundTruthError(f"В {path} нет объекта scenarios")

    expected: dict[CellKey, ExpectedCell] = {}
    for scenario, payload in document["scenarios"].items():
        covenants = payload.get("covenants") if isinstance(payload, dict) else None
        if not isinstance(covenants, dict):
            raise GroundTruthError(f"{path}: у сценария {scenario} нет ковенантов")
        for covenant, cell in covenants.items():
            expected[scenario, covenant] = ExpectedCell(
                status=str(cell["status"]),
                actual=Decimal(str(cell["actual"])),
                evidence_txn_id=cell["evidence_txn_id"],
            )
    return expected


def accuracy(actual: Decimal | None, expected: Decimal) -> Decimal:
    """Доля сохранённых баллов за число: 1.0 при точном попадании, 0.0 при 5%."""
    if actual is None:
        return Decimal(0)
    if expected == 0:
        return Decimal(1) if actual == 0 else Decimal(0)
    error = abs(actual - expected) / abs(expected)
    return max(Decimal(0), Decimal(1) - error / TOLERANCE)


def score_cell(
    scenario: str,
    covenant: str,
    expected: ExpectedCell,
    answer: dict[str, Any] | None,
) -> CellScore:
    def empty(note: str) -> CellScore:
        return CellScore(
            scenario=scenario,
            covenant=covenant,
            expected=expected,
            status=None,
            actual=None,
            evidence_txn_id=None,
            status_points=Decimal(0),
            actual_points=Decimal(0),
            evidence_points=Decimal(0),
            note=note,
        )

    if answer is None:
        return empty("ячейка пропущена")

    status = answer.get("status")
    raw_actual = answer.get("actual")
    actual = Decimal(str(raw_actual)) if isinstance(raw_actual, int | float) else None
    evidence = answer.get("evidence_txn_id")

    if status != expected.status:
        return CellScore(
            scenario=scenario,
            covenant=covenant,
            expected=expected,
            status=status if isinstance(status, str) else None,
            actual=actual,
            evidence_txn_id=evidence if isinstance(evidence, str) else None,
            status_points=Decimal(0),
            actual_points=Decimal(0),
            evidence_points=Decimal(0),
            note=f"вердикт {status!r} вместо {expected.status!r} — вся ячейка обнулена",
        )

    share = accuracy(actual, expected.actual)
    # Когда в ключе улики нет, её 0.20 не выдаются даром: они убывают по той же
    # шкале, что и число. Поэтому точность actual весит вдвое больше на таких ячейках.
    if expected.evidence_txn_id is None:
        evidence_points = EVIDENCE_WEIGHT * share
        note = "" if share == 1 else "число неточно, улика в ключе пустая — теряются оба веса"
    else:
        hit = evidence == expected.evidence_txn_id
        evidence_points = EVIDENCE_WEIGHT if hit else Decimal(0)
        note = "" if hit else f"улика {evidence!r} вместо {expected.evidence_txn_id!r}"

    if actual is None:
        note = "actual отсутствует или не число"

    return CellScore(
        scenario=scenario,
        covenant=covenant,
        expected=expected,
        status=status,
        actual=actual,
        evidence_txn_id=evidence if isinstance(evidence, str) else None,
        status_points=STATUS_WEIGHT,
        actual_points=ACTUAL_WEIGHT * share,
        evidence_points=evidence_points,
        note=note,
    )


def score_submission(submission: Submission, expected: dict[CellKey, ExpectedCell]) -> ScoreReport:
    """Оценивает ровно то, что уйдёт в файл, — со всеми округлениями сериализации."""
    document = submission.model_dump(mode="json")
    answers: dict[str, dict[str, Any]] = document["answers"]

    cells = [
        score_cell(scenario, covenant, cell, answers.get(scenario, {}).get(covenant))
        for (scenario, covenant), cell in expected.items()
    ]
    provided = submission.cell_keys()
    return ScoreReport(
        cells=tuple(cells),
        missing=tuple(sorted(set(expected) - provided)),
        unexpected=tuple(sorted(provided - set(expected))),
    )
