"""Сквозной прогон на англоязычном наборе: от документов до Submission.json.

Проверяется не арифметика — она разобрана в модульных тестах, — а то, что звенья
сходятся: перепись отдаёт документы компилятору, договор задаёт период, примечания
дают одну величину сами, вторую дочитывает resolver, классификация доносит статьи до
селекторов, а инварианты и шаблон пропускают результат наружу.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from halyk.cli import app
from halyk.hashing import sha256_file
from halyk.models.manifest import RunMode, RunStatus
from halyk.output.template import SubmissionTemplate
from halyk.output.validator import validate_file
from halyk.pipeline import Engines, solve
from halyk.pipeline.solve import SolveResult
from halyk.run.context import RunContext
from halyk.run.trace import read_lineage

SCHEMAS = Path("schemas")


@pytest.fixture
def completed(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> tuple[SolveResult, RunContext, Engines]:
    context = make_context()
    engines = make_engines(context)
    result = solve(context, dataset, tmp_path / "Submission.json", engines=engines)
    return result, context, engines


def answers(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    document: dict[str, dict[str, dict[str, dict[str, object]]]] = json.loads(
        path.read_text(encoding="utf-8")
    )
    return document["answers"]


def test_every_template_cell_is_answered(
    completed: tuple[SolveResult, RunContext, Engines], dataset: Path
) -> None:
    result, _, _ = completed
    filled = answers(result.submission_path)
    assert sorted((scenario, clause) for scenario, cells in filled.items() for clause in cells) == [
        ("E1", "6.1"),
        ("E1", "6.2"),
        ("E2", "6.1"),
        ("E2", "6.2"),
        ("E2", "6.3"),
    ]
    assert all(
        cell["status"] is not None and cell["actual"] is not None
        for cells in filled.values()
        for cell in cells.values()
    )


def test_related_party_covenant_is_breached_with_a_single_culprit(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """250 000 и 30 000 связанной стороне против потолка 240 000.

    Уликой признаётся только строка на 250 000: без неё остаётся 30 000 и нарушения
    нет, а без второй строки нарушение сохраняется. Изъятая по раскрытию операция на
    500 000 в расчёт не входит вовсе.
    """
    result, _, _ = completed
    cell = answers(result.submission_path)["E1"]["6.1"]
    assert cell["status"] == "BREACH"
    assert Decimal(str(cell["actual"])) == Decimal("280000.00")
    assert cell["evidence_txn_id"] == "TXN-E1-0001"


def test_fact_from_the_deterministic_layer_reaches_the_formula(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """Операционные расходы 120 000 плюс обязательство 120 000 из примечаний."""
    result, _, _ = completed
    cell = answers(result.submission_path)["E1"]["6.2"]
    assert cell["status"] == "COMPLIANT"
    assert Decimal(str(cell["actual"])) == Decimal("240000.00")


def test_fact_read_by_the_resolver_reaches_the_formula(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """Коммунальные 90 000 плюс гарантия 45 000, которой детерминированный слой не знал."""
    result, _, _ = completed
    cell = answers(result.submission_path)["E2"]["6.1"]
    assert cell["status"] == "COMPLIANT"
    assert Decimal(str(cell["actual"])) == Decimal("135000.00")


def test_metric_without_its_own_fact_model_reaches_the_formula(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """EBITDA не входит в три специализированных вида и доезжает общей величиной."""
    result, _, _ = completed
    cell = answers(result.submission_path)["E2"]["6.3"]
    assert cell["status"] == "COMPLIANT"
    assert Decimal(str(cell["actual"])) == Decimal("300000.00")


def test_resolver_is_asked_only_where_something_is_missing(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """У первого заёмщика все требования закрыл разбор документов — спрашивать нечего."""
    _, _, engines = completed
    assert engines.compiler.runner.send.accounts == ["ACC-4001", "ACC-4002"]  # type: ignore[attr-defined]
    assert engines.resolver.runner.send.accounts == ["ACC-4002"]  # type: ignore[attr-defined]


def test_batches_are_one_per_borrower(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    result, _, _ = completed
    batches = {
        report.scenario_id: (report.compiler_batches, report.resolver_batches)
        for report in result.report.borrowers
    }
    assert batches == {"E1": (1, 0), "E2": (1, 1)}


def test_submission_matches_the_schema_and_the_template(
    completed: tuple[SolveResult, RunContext, Engines], dataset: Path
) -> None:
    result, _, _ = completed
    from halyk.models.submission import Submission  # noqa: PLC0415

    assert validate_file(result.submission_path, "submission", SCHEMAS) == []
    parsed = Submission.model_validate_json(result.submission_path.read_text(encoding="utf-8"))
    template = SubmissionTemplate.load(dataset / "submission_template.json")
    assert template.check(parsed) == []


def test_every_artifact_is_on_disk(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    result, context, _ = completed
    for path in (
        context.submission_path,
        context.lineage_path,
        context.metrics_path,
        context.manifest_path,
        context.report_path,
    ):
        assert path.exists(), path
    assert context.submission_path.read_text(encoding="utf-8") == result.submission_path.read_text(
        encoding="utf-8"
    )


def test_lineage_explains_every_answer(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    _, context, _ = completed
    records = read_lineage(context.lineage_path)
    assert len(records) == 5
    breach = next(r for r in records if r.borrower_id == "ACC-4001" and r.covenant_id == "6.1")
    assert breach.measured_value == Decimal("280000.00")
    assert breach.threshold == Decimal("240000")
    assert breach.evidence_transaction_id == "TXN-E1-0001"
    assert breach.source_refs[0].file_name == "e1-agreement.pdf"
    assert all(check.passed for record in records for check in record.invariants)


def test_manifest_names_every_input(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    """Датасет, шаблон, промпты, модели, результат и кэш — всё отпечатками."""
    result, context, _ = completed
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == RunStatus.COMPLETED.value
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["template_sha256"]) == 64
    assert set(manifest["prompts_sha256"]) == {"compiler", "resolver", "classifier", "ocr"}
    assert set(manifest["models"]) == {"covenant_compiler", "ocr", "classifier", "verifier"}
    # Отпечаток обязан сходиться с файлом на диске, а не с тем, что мы собирались
    # записать: на Windows перевод строки по умолчанию другой, и это ровно та ошибка,
    # которую заметит только проверяющий.
    assert manifest["submission_sha256"] == sha256_file(result.submission_path)
    assert len(manifest["model_cache_sha256"]) == 64


def test_metrics_are_split_by_stage(
    completed: tuple[SolveResult, RunContext, Engines],
) -> None:
    _, context, _ = completed
    metrics = json.loads(context.metrics_path.read_text(encoding="utf-8"))
    stages = metrics["stages"]
    assert set(stages) == {"ocr", "compiler", "resolver", "classifier", "verifier"}
    assert stages["compiler"]["calls"] == 2
    assert stages["compiler"]["live_calls"] == 2
    assert stages["compiler"]["tokens_in"] > 0
    assert stages["resolver"]["calls"] == 1
    assert stages["classifier"]["calls"] >= 2
    assert stages["ocr"]["calls"] == 1
    assert metrics["ocr_pages"] == 1
    assert metrics["covenants_answered"] == metrics["covenants_found"] == 5
    assert metrics["invariants_failed"] == 0
    assert metrics["invariants_passed"] == 15


def test_report_is_clean(completed: tuple[SolveResult, RunContext, Engines]) -> None:
    _, context, _ = completed
    report = json.loads(context.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["problems"] == []
    assert report["template"] == {"cells": 5, "answered": 5, "missing": []}


def test_cli_validate_accepts_the_answer(
    completed: tuple[SolveResult, RunContext, Engines], dataset: Path
) -> None:
    """Проверка тем же способом, каким её сделают организаторы."""
    checked = CliRunner().invoke(
        app,
        [
            "validate",
            "--submission",
            str(completed[0].submission_path),
            "--template",
            str(dataset / "submission_template.json"),
        ],
    )
    assert checked.exit_code == 0, checked.output


def test_cli_stops_before_the_run_without_the_answer_header(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Команда и почта спрашиваются до первого вызова, а не после оплаченного прогона."""
    monkeypatch.setenv("HALYK_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("HALYK_TEAM", "")
    monkeypatch.setenv("HALYK_CONTACT_EMAIL", "")
    started = CliRunner().invoke(
        app,
        ["solve", "--input", str(dataset), "--output", str(tmp_path / "Submission.json")],
    )
    assert started.exit_code == 1
    assert "HALYK_TEAM" in started.output


def test_dataset_has_no_ground_truth(dataset: Path) -> None:
    """Расчётный путь обязан работать при физическом отсутствии ключа."""
    assert list(dataset.rglob("ground_truth.json")) == []


def test_second_run_repeats_the_answer_from_the_cache(
    completed: tuple[SolveResult, RunContext, Engines],
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Повтор по тому же кэшу: ни одного живого вызова и тот же файл до байта."""
    result, _, _ = completed
    first = result.submission_path.read_text(encoding="utf-8")

    context = make_context(mode=RunMode.RESUME)
    engines = make_engines(context, compiler={}, resolver={})
    repeated = solve(context, dataset, tmp_path / "Repeated.json", engines=engines)

    assert repeated.submission_path.read_text(encoding="utf-8") == first
    assert engines.compiler.runner.send.sent == []  # type: ignore[attr-defined]
    assert engines.resolver.runner.send.sent == []  # type: ignore[attr-defined]
    assert engines.classifier.model.live_calls == 0
    assert engines.ocr is not None and engines.ocr.live_calls == 0
    assert repeated.metrics.live_calls == 0
