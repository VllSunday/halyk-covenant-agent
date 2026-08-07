"""Шаблон ответа как источник истины по составу ячеек.

Набор заёмщиков и пунктов берётся из файла организаторов, а не из нашего кода:
пропущенная ячейка стоит столько же, сколько неверная, а лишняя не засчитывается.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halyk.models.submission import Submission
from halyk.output.validator import ValidationIssue


class TemplateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SubmissionTemplate:
    cells: tuple[tuple[str, str], ...]

    @classmethod
    def load(cls, path: Path) -> SubmissionTemplate:
        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TemplateError(f"{path} — не JSON: {exc}") from exc

        if not isinstance(document, dict) or not isinstance(document.get("answers"), dict):
            raise TemplateError(f"В {path} нет объекта answers")

        cells: list[tuple[str, str]] = []
        for scenario, covenants in document["answers"].items():
            if not isinstance(covenants, dict):
                raise TemplateError(f"{path}: у сценария {scenario} ковенанты не объект")
            cells.extend((scenario, covenant) for covenant in covenants)

        if not cells:
            raise TemplateError(f"В {path} нет ни одной ячейки")
        return cls(cells=tuple(cells))

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(scenario for scenario, _ in self.cells))

    def key_set(self) -> set[tuple[str, str]]:
        return set(self.cells)

    def check(self, submission: Submission) -> list[ValidationIssue]:
        """Сверяет состав ячеек и заполненность верхнеуровневых полей."""
        issues = [
            ValidationIssue(location=field, message="поле не заполнено")
            for field in ("team", "contact_email", "model")
            if not getattr(submission, field).strip()
        ]

        expected, actual = self.key_set(), submission.cell_keys()
        issues.extend(
            ValidationIssue(location=f"answers/{scenario}/{covenant}", message="ячейка пропущена")
            for scenario, covenant in sorted(expected - actual)
        )
        issues.extend(
            ValidationIssue(
                location=f"answers/{scenario}/{covenant}", message="ячейки нет в шаблоне"
            )
            for scenario, covenant in sorted(actual - expected)
        )
        return issues
