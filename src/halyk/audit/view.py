"""Прогон, прочитанный с диска.

Читаются артефакты, а не пересчитывается работа: аудит обязан говорить о том прогоне,
который был, — в том числе о чужом, снятом на приватных данных, куда мы уже не
попадём.

Ячейка собирается из двух источников. Отчёт держит состав и итог, аудиторский след —
происхождение: адреса, отпечаток дерева, строки расчёта и инварианты. Ключ у них общий:
счёт заёмщика и номер пункта.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from halyk.hashing import sha256_payload
from halyk.models.lineage import LineageRecord
from halyk.run.context import (
    CACHE_INDEX_NAME,
    LINEAGE_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    REPORT_NAME,
)
from halyk.run.trace import read_lineage


class AuditError(RuntimeError):
    """Каталог прогона нельзя прочитать: нет артефактов или они не разбираются."""


@dataclass(frozen=True, slots=True)
class CellView:
    """Одна ячейка прогона со всем, что о ней известно."""

    scenario_id: str
    clause_id: str
    account_id: str
    status: str
    actual: Decimal
    threshold: Decimal
    evidence_txn_id: str | None
    confidence: float
    rows: int
    ir_hash: str = ""
    sources: tuple[str, ...] = ()
    contributing_rows: tuple[str, ...] = ()
    failed_invariants: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def cell(self) -> tuple[str, str]:
        return self.scenario_id, self.clause_id

    @property
    def name(self) -> str:
        return f"{self.scenario_id}/{self.clause_id}"


@dataclass(frozen=True, slots=True)
class RunView:
    """Артефакты одного прогона в разобранном виде."""

    path: Path
    run_id: str
    mode: str
    status: str
    cells: dict[tuple[str, str], CellView]
    template_cells: int
    missing_cells: tuple[str, ...]
    problems: tuple[dict[str, str], ...]
    notes: tuple[dict[str, str], ...]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    cache_entries: tuple[dict[str, str], ...] = ()
    last_known_good: dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> int:
        return len(self.cells)

    @property
    def coverage(self) -> float:
        return self.answered / self.template_cells if self.template_cells else 0.0

    def problems_with(self, prefix: str) -> tuple[dict[str, str], ...]:
        return tuple(item for item in self.problems if item["code"].startswith(prefix))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuditError(f"В прогоне нет {path.name}: {path.parent}")
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(f"{path} не разбирается: {exc}") from exc
    return loaded


def _lineage(run: Path) -> dict[tuple[str, str], LineageRecord]:
    path = run / LINEAGE_NAME
    if not path.exists():
        return {}
    return {(record.borrower_id, record.covenant_id): record for record in read_lineage(path)}


def _cell(
    scenario: str, account: str, raw: dict[str, Any], trace: LineageRecord | None
) -> CellView:
    return CellView(
        scenario_id=scenario,
        clause_id=str(raw["clause_id"]),
        account_id=account,
        status=str(raw["status"]),
        actual=Decimal(str(raw["actual"])),
        threshold=Decimal(str(raw["threshold"])),
        evidence_txn_id=raw.get("evidence_txn_id"),
        confidence=float(raw.get("confidence", 0.0)),
        rows=int(raw.get("rows", 0)),
        ir_hash=trace.ir_hash if trace else "",
        sources=tuple(f"{ref.file_name}:{ref.page}" for ref in trace.source_refs) if trace else (),
        contributing_rows=trace.contributing_row_ids if trace else (),
        failed_invariants=(
            tuple(check.name for check in trace.failed_invariants()) if trace else ()
        ),
        diagnostics=trace.notes if trace else (),
    )


def load(run: Path) -> RunView:
    """Прочитать каталог прогона. Отсутствие отчёта — отказ, а не пустой аудит."""
    if not run.is_dir():
        raise AuditError(f"{run} — не каталог прогона")

    report = _load_json(run / REPORT_NAME)
    traces = _lineage(run)
    cells: dict[tuple[str, str], CellView] = {}
    for borrower in report.get("borrowers", []):
        scenario, account = borrower["scenario_id"], borrower["account_id"]
        for raw in borrower.get("cells", []):
            view = _cell(scenario, account, raw, traces.get((account, str(raw["clause_id"]))))
            cells[view.cell] = view

    template = report.get("template", {})
    index = run / CACHE_INDEX_NAME
    return RunView(
        path=run,
        run_id=str(report.get("run_id", run.name)),
        mode=str(report.get("mode", "")),
        status=str(report.get("status", "")),
        cells=cells,
        template_cells=int(template.get("cells", 0)),
        missing_cells=tuple(template.get("missing", [])),
        problems=tuple(report.get("problems", [])),
        notes=tuple(report.get("notes", [])),
        metrics=_load_json(run / METRICS_NAME) if (run / METRICS_NAME).exists() else {},
        manifest=_load_json(run / MANIFEST_NAME) if (run / MANIFEST_NAME).exists() else {},
        cache_entries=tuple(_load_json(index).get("entries", [])) if index.exists() else (),
        last_known_good=report.get("last_known_good", {}),
    )


@dataclass(frozen=True, slots=True)
class ReplayCheck:
    """Хватит ли общего кэша, чтобы повторить прогон, не спрашивая моделей."""

    total: int
    present: int
    missing: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def deterministic(self) -> bool:
        return not self.missing and not self.changed


def check_replay(view: RunView, cache_root: Path) -> ReplayCheck:
    """Проверка по содержимому, а не по наличию файла.

    Запись с тем же именем и другим содержимым — это другой ответ модели, и повтор
    дал бы не тот же сабмит. Отличить одно от другого можно только отпечатком.
    """
    missing: list[str] = []
    changed: list[str] = []
    for entry in view.cache_entries:
        path = cache_root / entry["role"] / f"{entry['key']}.json"
        if not path.exists():
            missing.append(f"{entry['role']}/{entry['key'][:12]}")
            continue
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            changed.append(f"{entry['role']}/{entry['key'][:12]}")
            continue
        if sha256_payload(stored) != entry["sha256"]:
            changed.append(f"{entry['role']}/{entry['key'][:12]}")
    return ReplayCheck(
        total=len(view.cache_entries),
        present=len(view.cache_entries) - len(missing),
        missing=tuple(missing),
        changed=tuple(changed),
    )
