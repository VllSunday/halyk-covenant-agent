"""Разбор одного ответа в терминале. Читает lineage.jsonl, ничего не пересчитывает."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from halyk.models.lineage import LineageRecord
from halyk.models.result import Verdict

_OPERATOR_SIGNS = {
    "ge": "≥",
    "gt": ">",
    "le": "≤",
    "lt": "<",
    "eq": "=",
    "ne": "≠",
}


def render_explanation(record: LineageRecord) -> RenderableType:
    return Group(
        _header(record),
        _sources(record),
        _calculation(record),
        _conclusion(record),
        _invariants(record),
    )


def _header(record: LineageRecord) -> RenderableType:
    title = f"{record.borrower_id} / {record.covenant_id}"
    body = Text.assemble(
        ("Пункт: ", "dim"),
        f"{record.clause_id}\n",
        ("Калькулятор: ", "dim"),
        f"{record.calculator}\n",
        ("Уверенность компиляции: ", "dim"),
        f"{record.ir_confidence:.2f}",
    )
    return Panel(body, title=title, title_align="left")


def _sources(record: LineageRecord) -> RenderableType:
    table = Table(title="Источники", title_justify="left", show_edge=False, pad_edge=False)
    table.add_column("Документ")
    table.add_column("Стр.", justify="right")
    table.add_column("Цитата", overflow="fold")
    for ref in record.source_refs:
        table.add_row(ref.file_name, str(ref.page), ref.quote or "—")
    return table


def _calculation(record: LineageRecord) -> RenderableType:
    table = Table(title="Расчёт", title_justify="left", show_edge=False, pad_edge=False)
    table.add_column("Шаг")
    table.add_column("Строк до", justify="right")
    table.add_column("Строк после", justify="right")
    for step in record.steps:
        table.add_row(step.description, str(step.rows_before), str(step.rows_after))
    table.add_row(
        "[dim]участвовало в результате[/dim]",
        "",
        str(len(record.contributing_row_ids)),
    )
    return table


def _conclusion(record: LineageRecord) -> RenderableType:
    sign = _OPERATOR_SIGNS.get(record.operator, record.operator)
    violated = record.verdict is Verdict.VIOLATED
    body = Text.assemble(
        (f"{record.measured_value} ", "bold"),
        (f"{sign} {record.threshold}\n", "dim"),
        ("нарушен" if violated else "соблюдён", "bold red" if violated else "bold green"),
    )
    if record.evidence_transaction_id:
        body.append(f"\nУлика: {record.evidence_transaction_id}")
    elif violated:
        body.append("\nУлика: не выделяется, нарушение агрегатное", style="dim")
    return Panel(body, title="Вывод", title_align="left")


def _invariants(record: LineageRecord) -> RenderableType:
    failed = record.failed_invariants()
    if not failed:
        return Text(f"Инварианты: пройдено {len(record.invariants)}", style="dim")
    table = Table(title="Инварианты не прошли", title_justify="left", show_edge=False)
    table.add_column("Проверка")
    table.add_column("Детали", overflow="fold")
    for check in failed:
        table.add_row(check.name, check.detail or "")
    return table
