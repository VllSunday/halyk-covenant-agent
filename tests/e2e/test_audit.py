"""Разбор готового прогона и сравнение двух прогонов.

Аудит читает артефакты и ничего не пересчитывает: он обязан работать на чужом прогоне,
снятом на приватных данных, куда мы уже не попадём. Поэтому и проверяется он на том,
что осталось на диске, а не на объектах в памяти.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from halyk.audit import Grade, check_replay, compare, counts, grade_run, load
from halyk.cli import app
from halyk.models.formula import Constant
from halyk.pipeline import Engines, PipelineError, solve
from halyk.run.context import RunContext

from .conftest import E1_ACCOUNT, E2_ACCOUNT, compiled, e1_clauses, e2_clauses


def complete(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
    run_id: str = "good",
) -> RunContext:
    context = make_context(run_id=run_id)
    solve(context, dataset, tmp_path / f"{run_id}.json", engines=make_engines(context))
    return context


def broken(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
    run_id: str = "broken",
) -> RunContext:
    """Прогон, у которого не собрался первый заёмщик."""
    context = make_context(run_id=run_id, fresh=True)
    answers = {E1_ACCOUNT: [compiled(e1_clauses()[:1])], E2_ACCOUNT: [compiled(e2_clauses())]}
    with pytest.raises(PipelineError):
        solve(
            context,
            dataset,
            tmp_path / f"{run_id}.json",
            engines=make_engines(context, compiler=answers),
        )
    return context


def test_audit_reads_a_finished_run(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    context = complete(dataset, tmp_path, make_context, make_engines)
    view = load(context.root)

    assert view.status == "completed"
    assert view.answered == view.template_cells == 5
    assert view.coverage == 1.0
    assert view.cells["E1", "6.1"].evidence_txn_id == "TXN-E1-0001"
    assert view.cells["E1", "6.1"].sources == ("e1-agreement.pdf:1",)
    assert view.cells["E1", "6.1"].contributing_rows == ("TXN-E1-0001", "TXN-E1-0002")


def test_every_answered_cell_is_green(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Улика у нарушения есть, инварианты прошли, уверенность высокая."""
    context = complete(dataset, tmp_path, make_context, make_engines)
    graded = grade_run(load(context.root))
    assert counts(graded) == {"GREEN": 5, "YELLOW": 0, "RED": 0}


def test_unanswered_cells_are_red_with_a_reason(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    context = broken(dataset, tmp_path, make_context, make_engines)
    graded = {item.name: item for item in grade_run(load(context.root))}

    assert counts(list(graded.values())) == {"GREEN": 3, "YELLOW": 0, "RED": 2}
    assert graded["E1/6.1"].grade is Grade.RED
    assert any("borrower_failed" in reason for reason in graded["E1/6.1"].reasons)


def test_replay_needs_every_cache_entry(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Повтор считается возможным по содержимому записей, а не по их наличию."""
    context = complete(dataset, tmp_path, make_context, make_engines)
    view = load(context.root)
    cache_root = context.settings.artifacts_dir / "cache"

    assert check_replay(view, cache_root).deterministic

    entry = view.cache_entries[0]
    (cache_root / entry["role"] / f"{entry['key']}.json").write_text("{}", encoding="utf-8")
    spoiled = check_replay(view, cache_root)
    assert not spoiled.deterministic
    assert len(spoiled.changed) == 1


def test_compare_finds_nothing_between_identical_runs(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    first = complete(dataset, tmp_path, make_context, make_engines, run_id="first")
    second = complete(dataset, tmp_path, make_context, make_engines, run_id="second")

    result = compare(load(first.root), load(second.root))
    assert result.changed == ()
    assert not result.has_regression


def test_compare_calls_a_lost_cell_a_regression(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    baseline = complete(dataset, tmp_path, make_context, make_engines)
    candidate = broken(dataset, tmp_path, make_context, make_engines)

    result = compare(load(baseline.root), load(candidate.root))
    assert result.has_regression
    assert [diff.name for diff in result.regressions] == ["E1/6.1", "E1/6.2"]
    assert result.counts["RED"] == (0, 2)


def test_compare_shows_a_changed_number_without_calling_it_a_regression(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Ради изменившегося числа правки и делаются, но в отчёте оно обязано быть."""
    baseline = complete(dataset, tmp_path, make_context, make_engines, run_id="baseline")

    # Пункт перекомпилирован с другим порогом: ячейка остаётся соблюдённой, но и
    # порог, и отпечаток дерева стали другими.
    clauses = list(e1_clauses())
    formula = clauses[1].formula.model_copy(update={"threshold": Constant(value=Decimal("500000"))})
    clauses[1] = clauses[1].model_copy(update={"formula": formula})

    candidate = make_context(run_id="candidate", fresh=True)
    answers = {E1_ACCOUNT: [compiled(tuple(clauses))], E2_ACCOUNT: [compiled(e2_clauses())]}
    solve(
        candidate,
        dataset,
        tmp_path / "candidate.json",
        engines=make_engines(candidate, compiler=answers),
    )

    result = compare(load(baseline.root), load(candidate.root))
    assert not result.has_regression
    changed = {diff.name: diff.changes for diff in result.changed}
    assert set(changed) == {"E1/6.2"}
    assert any("порог: 700000 → 500000" in item for item in changed["E1/6.2"])
    assert any(item.startswith("дерево:") for item in changed["E1/6.2"])


def test_cli_audit_and_compare(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    baseline = complete(dataset, tmp_path, make_context, make_engines, run_id="baseline")
    candidate = broken(dataset, tmp_path, make_context, make_engines)

    clean = CliRunner().invoke(app, ["audit", "--run", str(baseline.root), "--all"])
    assert clean.exit_code == 0, clean.output
    assert "GREEN" in clean.output

    failed = CliRunner().invoke(app, ["audit", "--run", str(candidate.root)])
    assert failed.exit_code == 1

    regression = CliRunner().invoke(
        app,
        ["compare", "--baseline", str(baseline.root), "--candidate", str(candidate.root)],
    )
    assert regression.exit_code == 1
    assert "E1/6.1" in regression.output
