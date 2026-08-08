"""Чем заканчивается неполный прогон.

Проверяется одно свойство на все случаи: неполный ответ обязан выглядеть как ошибка.
Не как Submission.json с `null` в ячейке, не как зелёный код возврата и не как тихо
пропавшая ячейка — ноль баллов в каждом из этих случаев одинаковый, а заметить успеешь
только первый.

Диагностика при этом остаётся на диске, включая ту её часть, что относится к
заёмщикам, у которых всё сошлось.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.execution.executor import Outcome
from halyk.models.manifest import RunMode, RunStatus
from halyk.output.template import SubmissionTemplate
from halyk.pipeline import Engines, PipelineError, build_engines, solve
from halyk.pipeline.submission import CompositionError, IncompleteCellError, build_submission
from halyk.run.context import RunContext

from .conftest import (
    E1_ACCOUNT,
    E2_ACCOUNT,
    TEMPLATE,
    compiled,
    e1_clauses,
    e2_clauses,
    prime_ocr_cache,
    resolved,
    resolved_ebitda,
    resolved_guarantee,
)

REFUSAL = {
    "requirement_id": "e2-guarantee",
    "reason": "not_found",
    "explanation": "в примечаниях этой величины нет",
    "candidate_source_refs": (),
}


def refused_guarantee() -> dict[str, Any]:
    """Отказ по одному требованию при закрытом втором: батч обязан ответить по каждому."""
    return resolved(facts=(resolved_ebitda(),), unresolved_requirements=(REFUSAL,))


def run(context: RunContext, dataset: Path, tmp_path: Path, engines: Engines) -> dict[str, Any]:
    """Прогон, который обязан упасть. Возвращает отчёт, прочитанный с диска."""
    with pytest.raises(PipelineError):
        solve(context, dataset, tmp_path / "Submission.json", engines=engines)

    assert not (tmp_path / "Submission.json").exists()
    assert not context.submission_path.exists()
    report: dict[str, Any] = json.loads(context.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    return report


def codes(report: dict[str, Any]) -> set[str]:
    return {item["code"] for item in report["problems"]}


def outcome(scenario: str, clause: str, *, empty: bool = False) -> Outcome:
    return Outcome(
        scenario_id=scenario,
        clause_id=clause,
        actual=None if empty else Decimal("1.00"),
        status=None if empty else "COMPLIANT",
        evidence_txn_id=None,
        rows=(),
        facts=(),
        diagnostics=(),
    )


def test_compiler_skipped_a_cell(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Пропущенная компилятором ячейка не превращается в пустую ячейку ответа."""
    context = make_context()
    answers = {E1_ACCOUNT: [compiled(e1_clauses()[:1])], E2_ACCOUNT: [compiled(e2_clauses())]}
    report = run(context, dataset, tmp_path, make_engines(context, compiler=answers))

    assert "borrower_failed" in codes(report)
    assert "cell_not_answered" in codes(report)
    # Код отказа компилятора виден в отчёте: наружу поднимается только «не разобрался».
    assert any("cell_is_missing" in note["detail"] for note in report["notes"])


def test_resolver_refused_the_requirement(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Названный отказ — законный ответ модели и незакрытая ячейка одновременно."""
    context = make_context()
    engines = make_engines(context, resolver={E2_ACCOUNT: [refused_guarantee()]})
    report = run(context, dataset, tmp_path, engines)

    assert "requirement_is_open" in codes(report)
    assert any("not_found" in item["detail"] for item in report["problems"])
    answered = {item["scenario_id"]: len(item["cells"]) for item in report["borrowers"]}
    assert answered == {"E1": 2, "E2": 2}


def test_two_values_for_one_requirement_resolve_nothing(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Спор источников не разрешается выбором первого попавшегося."""
    context = make_context()
    disputed = resolved(
        facts=(resolved_guarantee(), resolved_guarantee("52000"), resolved_ebitda())
    )
    engines = make_engines(context, resolver={E2_ACCOUNT: [disputed]})

    assert "requirement_is_ambiguous" in codes(run(context, dataset, tmp_path, engines))


def test_one_broken_borrower_keeps_the_diagnostics_of_the_others(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    context = make_context()
    original = e1_clauses()[0]
    alien = original.model_copy(
        update={"formula": original.formula.model_copy(update={"clause_id": "9.9"})}
    )
    answers = {E1_ACCOUNT: [compiled((alien,))], E2_ACCOUNT: [compiled(e2_clauses())]}
    report = run(context, dataset, tmp_path, make_engines(context, compiler=answers))

    healthy = next(item for item in report["borrowers"] if item["scenario_id"] == "E2")
    assert len(healthy["cells"]) == 3
    assert healthy["problems"] == []
    assert healthy["resolver_batches"] == 1


def test_budget_stops_live_calls_for_parallel_borrowers(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Потолок проверяется до сети у каждой уже запущенной параллельной задачи."""
    context = make_context(max_live_calls=0)
    report = run(context, dataset, tmp_path, make_engines(context))

    assert codes(report) >= {"budget_exhausted", "cell_not_answered"}
    assert report["budget"]["live_calls"] == 0


def test_offline_cache_miss_never_reaches_the_network(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
) -> None:
    """Пустой кэш в офлайне — остановка, а не тихий живой вызов.

    Роли здесь настоящие: подставная отправка доказала бы только то, что её не
    позвали. Конструктор сетевого клиента подменён взрывающимся на весь модуль.
    """
    # Распознавание оплачено раньше и лежит в общем кэше: иначе прогон остановился бы
    # на первой же странице и до вопроса к модели не дошёл.
    context = make_context(offline=True)
    prime_ocr_cache(context.settings, dataset)
    report = run(context, dataset, tmp_path, build_engines(context))

    assert any("CACHE_MISS" in item["detail"] for item in report["problems"])
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == RunStatus.FAILED.value


def test_template_scenario_without_transactions_stops_the_run(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Шаблон требует сценарий, которого нет в реестре: считать нечего и некому."""
    extended = json.loads(json.dumps(TEMPLATE))
    extended["answers"]["E9"] = {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}}
    template = tmp_path / "extended_template.json"
    template.write_text(json.dumps(extended, ensure_ascii=False), encoding="utf-8")

    context = make_context()
    with pytest.raises(PipelineError, match="E9"):
        solve(
            context,
            dataset,
            tmp_path / "Submission.json",
            template_path=template,
            engines=make_engines(context),
        )
    assert not (tmp_path / "Submission.json").exists()


def test_submission_refuses_a_composition_that_differs_from_the_template(
    dataset: Path, make_context: Callable[..., RunContext]
) -> None:
    template = SubmissionTemplate.load(dataset / "submission_template.json")
    answers = {
        ("E1", "6.1"): outcome("E1", "6.1"),
        ("E1", "6.2"): outcome("E1", "6.2"),
        ("E2", "6.1"): outcome("E2", "6.1"),
        ("E2", "6.2"): outcome("E2", "6.2"),
        ("E2", "6.3"): outcome("E2", "6.3"),
        ("E2", "6.4"): outcome("E2", "6.4"),
    }
    with pytest.raises(CompositionError, match=r"6\.4"):
        build_submission(template, make_context().settings, answers)


def test_cell_without_a_verdict_is_never_serialised(
    dataset: Path, make_context: Callable[..., RunContext]
) -> None:
    """`null` в ячейке стоит тех же нуля баллов, что и неверное число."""
    template = SubmissionTemplate.load(dataset / "submission_template.json")
    answers = {
        ("E1", "6.1"): outcome("E1", "6.1"),
        ("E1", "6.2"): outcome("E1", "6.2"),
        ("E2", "6.1"): outcome("E2", "6.1"),
        ("E2", "6.2"): outcome("E2", "6.2", empty=True),
        ("E2", "6.3"): outcome("E2", "6.3"),
    }
    with pytest.raises(IncompleteCellError, match=r"E2/6\.2"):
        build_submission(template, make_context().settings, answers)


def test_repeated_failing_run_fails_the_same_way(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Негодный ответ в кэш не ложится, и повтор по кэшу воспроизводит тот же отказ."""
    refusal = {E2_ACCOUNT: [refused_guarantee()]}
    context = make_context()
    first = run(context, dataset, tmp_path, make_engines(context, resolver=refusal))

    repeated = make_context(mode=RunMode.RESUME)
    second = run(repeated, dataset, tmp_path, make_engines(repeated, resolver=refusal))
    assert codes(second) == codes(first)
