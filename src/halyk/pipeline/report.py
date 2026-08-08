"""Диагностический отчёт прогона.

Пишется всегда — и когда ответ собрался, и особенно когда не собрался. Упавший
прогон без отчёта не отличить от незапущенного, а разбираться в нём придётся тогда,
когда времени уже нет.

Разделение на `Problem` и `Note` — не про тяжесть формулировки. Проблема означает
ячейку, которой не будет в ответе, и потому останавливает строгий прогон. Замечание
означает, что что-то стоит посмотреть глазами: неучтённую корректировку, операцию без
статьи, предупреждение инварианта. Ответ при этом собран, и валить его из-за замечания
значило бы менять полный ответ на никакой.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Problem:
    """То, из-за чего ячейка не получит ответа."""

    code: str
    subject: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Note:
    """То, что стоит посмотреть, но что ответа не отменяет."""

    code: str
    subject: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BorrowerReport:
    """Что вышло по одному заёмщику. Собирается и для тех, кто не досчитался."""

    scenario_id: str
    account_id: str
    period: str = ""
    transactions: int = 0
    compiler_batches: int = 0
    resolver_batches: int = 0
    cells: tuple[dict[str, Any], ...] = ()
    problems: tuple[Problem, ...] = ()
    notes: tuple[Note, ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "account_id": self.account_id,
            "period": self.period,
            "transactions": self.transactions,
            "compiler_batches": self.compiler_batches,
            "resolver_batches": self.resolver_batches,
            "cells": list(self.cells),
            "problems": [item.record() for item in self.problems],
            "notes": [item.record() for item in self.notes],
        }


@dataclass(slots=True)
class RunReport:
    """Отчёт целиком: заёмщики, инварианты, бюджет и состав ячеек."""

    run_id: str
    mode: str
    borrowers: list[BorrowerReport] = field(default_factory=list)
    invariants: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    template_cells: tuple[tuple[str, str], ...] = ()
    answered_cells: tuple[tuple[str, str], ...] = ()
    failures: list[Problem] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    # Что лежит в файле ответа сейчас. У упавшего прогона это предыдущий полный
    # ответ, и знать про него нужно до того, как решишь сдавать.
    last_known_good: dict[str, Any] = field(default_factory=dict)

    @property
    def problems(self) -> list[Problem]:
        """Все препятствия прогона: общие и пришедшие от заёмщиков."""
        return [*self.failures, *(item for report in self.borrowers for item in report.problems)]

    @property
    def is_clean(self) -> bool:
        return not self.problems

    @property
    def missing_cells(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(set(self.template_cells) - set(self.answered_cells)))

    def record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": "completed" if self.is_clean else "failed",
            "template": {
                "cells": len(self.template_cells),
                "answered": len(self.answered_cells),
                "missing": [f"{scenario}/{clause}" for scenario, clause in self.missing_cells],
            },
            "problems": [item.record() for item in self.problems],
            "notes": [item.record() for item in self.notes],
            "invariants": self.invariants,
            "last_known_good": self.last_known_good,
            "budget": self.budget,
            "borrowers": [report.record() for report in self.borrowers],
        }

    def summary(self) -> str:
        """Одна строка для терминала: чего именно не хватило."""
        codes = sorted({item.code for item in self.problems})
        return ", ".join(codes) if codes else "нарушений нет"


def first_lines(problems: Sequence[Problem], limit: int = 5) -> list[str]:
    """Начало списка проблем для терминала: остальное лежит в отчёте."""
    return [f"{item.code} — {item.subject}: {item.detail}" for item in problems[:limit]]
