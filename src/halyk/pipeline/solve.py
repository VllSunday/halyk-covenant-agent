"""Полный прогон: датасет на входе, Submission.json на выходе.

Порядок стадий:
  1. подготовка входа, шаблон ответа, перепись архива и хеши документов
  2. разбор PDF нативно, страницы с непригодным текстовым слоем — в OCR
  3. детерминированный слой: факты и корректировки из документов
  4. по каждому заёмщику: компиляция пунктов, реестр, дочитывание, статьи, расчёт
  5. инварианты прогона и сборка ответа по шаблону
  6. артефакты: ответ, аудиторский след, метрики, отчёт, удостоверение прогона

Прогон строгий. Открытое требование, спорная величина, несчитанная ячейка и
непройденный инвариант заканчивают его ошибкой, а не ответом с дырой: `null` в ячейке
стоит тех же нуля баллов, что и неверное число, но выглядит как результат.

Ошибка одного заёмщика не отменяет остальных. Каждый считается отдельно, а его отказ
становится записью в отчёте — иначе один неразобранный договор уносил бы с собой всю
диагностику прогона, ради которой отчёт и пишется.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from halyk.execution.executor import Outcome
from halyk.hashing import sha256_bytes, sha256_payload
from halyk.ingest.inventory import DatasetInventory, InventoryError, build_inventory
from halyk.ingest.ledger import LedgerError
from halyk.ingest.scenarios import ScenarioMappingError
from halyk.knowledge.facts import FactSet, build_facts
from halyk.llm.cache import CacheMissError
from halyk.llm.runner import BudgetError, Role
from halyk.models.lineage import LineageRecord
from halyk.models.manifest import RunMode, RunStatus
from halyk.models.metrics import RunMetrics, StageMetrics
from halyk.models.result import CalculationStep
from halyk.output.template import SubmissionTemplate
from halyk.pipeline import keeper
from halyk.pipeline import metrics as stage_metrics
from halyk.pipeline.borrower import AnsweredCell, BorrowerResult, solve_borrower
from halyk.pipeline.dataset import find_template, unpack
from halyk.pipeline.engines import Engines, build_engines, prompt_digests
from halyk.pipeline.report import BorrowerReport, Note, Problem, RunReport
from halyk.pipeline.submission import build_submission
from halyk.run.context import RunContext
from halyk.run.trace import LineageWriter
from halyk.verification.invariants import (
    Severity,
    Violation,
    check_coverage,
    check_documents,
    check_facts,
    summarise,
)

CALCULATOR = "formula-ast"


class PipelineError(RuntimeError):
    """Прогон не дал полного ответа. Отчёт к этому моменту уже на диске."""

    def __init__(self, message: str, report: RunReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class SolveResult:
    submission_path: Path
    metrics: RunMetrics
    report: RunReport
    run_dir: Path


def _require_header(context: RunContext) -> None:
    """Команда и почта спрашиваются до первого вызова модели.

    Ответ без них организаторы не примут, и узнавать об этом после оплаченного
    прогона поздно.
    """
    missing = [
        name
        for name, value in (
            ("HALYK_TEAM", context.settings.team),
            ("HALYK_CONTACT_EMAIL", context.settings.contact_email),
        )
        if not value.strip()
    ]
    if missing:
        raise PipelineError(
            f"Не заполнена шапка ответа: {', '.join(missing)}. Задайте её в .env — "
            f"без неё Submission.json не примут"
        )


def _clauses_by_scenario(template: SubmissionTemplate) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for scenario, clause in template.cells:
        grouped.setdefault(scenario, []).append(clause)
    return grouped


def _failed(scenario: str, account: str, problem: Problem) -> BorrowerResult:
    return BorrowerResult(scenario_id=scenario, account_id=account, problems=(problem,))


def _run_borrowers(
    template: SubmissionTemplate,
    inventory: DatasetInventory,
    facts: FactSet,
    engines: Engines,
    *,
    max_concurrency: int = 1,
) -> list[BorrowerResult]:
    """Независимые заёмщики параллельно, результат — в порядке шаблона."""
    clauses = _clauses_by_scenario(template)
    accounts = inventory.scenarios.scenario_to_account

    def run_one(scenario: str) -> BorrowerResult:
        account = accounts.get(scenario, "")
        try:
            return solve_borrower(
                scenario,
                account,
                clauses[scenario],
                inventory=inventory,
                facts=facts,
                engines=engines,
            )
        except BudgetError as exc:
            # Budget.authorise защищён общей блокировкой: параллельные задачи могут
            # завершить уже разрешённые вызовы, но новый вызов потолок не пересечёт.
            return _failed(scenario, account, Problem("budget_exhausted", scenario, str(exc)))
        except Exception as exc:  # отказ заёмщика — запись в отчёте, а не конец прогона
            problem = Problem("borrower_failed", scenario, f"{type(exc).__name__}: {exc}")
            return _failed(scenario, account, problem)

    workers = max(1, max_concurrency)
    if workers == 1:
        return [run_one(scenario) for scenario in template.scenarios]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="halyk-borrower") as pool:
        futures = {scenario: pool.submit(run_one, scenario) for scenario in template.scenarios}
        # Получаем результаты в порядке шаблона, а не в порядке завершения потоков.
        return [futures[scenario].result() for scenario in template.scenarios]


def _steps(cell: AnsweredCell, transactions: int) -> tuple[CalculationStep, ...]:
    outcome = cell.outcome
    steps = [
        CalculationStep(
            description="отбор строк расчёта по селекторам дерева",
            rows_before=transactions,
            rows_after=len(outcome.rows),
        )
    ]
    if outcome.trigger_rows:
        steps.append(
            CalculationStep(
                description="строки условия срабатывания пункта",
                rows_before=transactions,
                rows_after=len(outcome.trigger_rows),
            )
        )
    return tuple(steps)


def _lineage(context: RunContext, result: BorrowerResult, cell: AnsweredCell) -> LineageRecord:
    formula = cell.clause.formula
    outcome = cell.outcome
    notes = [f"сценарий {result.scenario_id}", *(f"факт {name}" for name in outcome.facts)]
    notes += [f"{kind.value}: {path}" for kind, path in outcome.diagnostics]
    if not outcome.triggered:
        notes.append("условие применения пункта не сработало")

    return LineageRecord(
        run_id=context.run_id,
        recorded_at=datetime.now(UTC),
        borrower_id=result.account_id,
        covenant_id=formula.clause_id,
        clause_id=formula.clause_id,
        clause_effective_from=cell.clause.period_start.isoformat(),
        source_refs=formula.source_refs,
        ir_hash=sha256_payload(formula.model_dump(mode="json")),
        ir_confidence=formula.confidence,
        calculator=CALCULATOR,
        steps=_steps(cell, result.transactions),
        contributing_row_ids=outcome.rows,
        # Отдельного списка нарушающих строк у дерева нет: нарушение агрегатное, и
        # единственная строка, про которую это можно сказать, — контрфактическая улика.
        violating_row_ids=(outcome.evidence_txn_id,) if outcome.evidence_txn_id else (),
        measured_value=cell.actual,
        threshold=cell.threshold,
        operator=formula.operator.value,
        verdict=cell.verdict,
        evidence_transaction_id=outcome.evidence_txn_id,
        invariants=cell.invariants,
        notes=tuple(notes),
    )


def _write_lineage(context: RunContext, results: Sequence[BorrowerResult]) -> None:
    """След пишется заново на каждый прогон: дописанный к прошлому он необъясним."""
    context.lineage_path.unlink(missing_ok=True)
    with LineageWriter(context.lineage_path) as writer:
        for result in results:
            for cell in result.answered:
                writer.write(_lineage(context, result, cell))


def _cell_record(cell: AnsweredCell) -> dict[str, Any]:
    return {
        "clause_id": cell.outcome.clause_id,
        "status": cell.verdict.value,
        "actual": str(cell.actual),
        "threshold": str(cell.threshold),
        "evidence_txn_id": cell.outcome.evidence_txn_id,
        "rows": len(cell.outcome.rows),
        "confidence": cell.clause.formula.confidence,
    }


def _borrower_report(result: BorrowerResult) -> BorrowerReport:
    return BorrowerReport(
        scenario_id=result.scenario_id,
        account_id=result.account_id,
        period=f"{result.period.start} - {result.period.end}" if result.period else "",
        transactions=result.transactions,
        compiler_batches=result.compiler_batches,
        resolver_batches=result.resolver_batches,
        cells=tuple(_cell_record(cell) for cell in result.answered),
        problems=result.problems,
        notes=result.notes,
    )


def _warnings(violations: Sequence[Violation]) -> list[Note]:
    return [
        Note(code=item.code, subject=item.subject, detail=item.message)
        for item in violations
        if item.severity is Severity.WARNING
    ]


def _rejections(engines: Engines) -> list[Note]:
    """Ответы моделей, не прошедшие разбор.

    Код отказа виден только здесь: наружу поднимается «ответ не разобрался ни с
    первого раза, ни с повтора», и без этой записи разбирать пришлось бы вслепую.
    """
    return [
        Note(
            code=(
                "provider_request_rejected"
                if call.provider_status is not None or call.provider_error_code is not None
                else "model_answer_rejected"
            ),
            subject=f"{call.role.value}/{call.account_id}, попытка {call.attempt}",
            detail=call.note,
        )
        for runner in (engines.compiler.runner, engines.resolver.runner)
        for call in runner.calls
        if not call.valid
    ]


def _errors(violations: Sequence[Violation]) -> list[Problem]:
    return [
        Problem(code=item.code, subject=item.subject, detail=item.message)
        for item in violations
        if item.severity is Severity.ERROR
    ]


def _count(results: Sequence[BorrowerResult], code: str, *, notes: bool = False) -> int:
    if notes:
        return sum(1 for item in results for note in item.notes if note.code == code)
    return sum(1 for item in results for problem in item.problems if problem.code == code)


def _stages(
    engines: Engines, inventory: DatasetInventory, results: Sequence[BorrowerResult]
) -> dict[str, StageMetrics]:
    open_requirements = sum(
        1
        for result in results
        for problem in result.problems
        if problem.code.startswith("requirement_is_")
    )
    unanswered = sum(len(result.problems) for result in results)
    return {
        "ocr": stage_metrics.from_ocr(inventory.ocr_calls),
        "compiler": stage_metrics.from_runner(
            engines.compiler.runner, Role.COMPILER, unresolved=unanswered
        ),
        "resolver": stage_metrics.from_runner(
            engines.resolver.runner, Role.RESOLVER, unresolved=open_requirements
        ),
        "classifier": stage_metrics.from_classifier(
            engines.classifier.model,
            unresolved=_count(results, "transaction_without_category", notes=True),
        ),
        "verifier": (
            stage_metrics.from_classifier(engines.classifier.verifier)
            if engines.classifier.verifier is not None
            else StageMetrics()
        ),
    }


def _metrics(
    inventory: DatasetInventory,
    template: SubmissionTemplate,
    results: Sequence[BorrowerResult],
    *,
    engines: Engines,
    violations: Sequence[Violation],
    seconds: float,
) -> RunMetrics:
    stages = _stages(engines, inventory, results)
    answered = [cell for result in results for cell in result.answered]
    return RunMetrics(
        stages=stages,
        documents_total=len(inventory.documents),
        pages_total=inventory.page_count,
        ocr_pages=len(inventory.ocr_pages),
        transactions_total=sum(result.transactions for result in results),
        borrowers_total=len(results),
        covenants_found=len(template.cells),
        covenants_answered=len(answered),
        invariants_passed=sum(1 for cell in answered for check in cell.invariants if check.passed),
        invariants_failed=len(_errors(violations)) + _count(results, "invariant_failed"),
        verifier_calls=stages["verifier"].calls,
        # Нарушение без выделенной улики: число и вердикт есть, а показать пальцем не
        # на что. Балл за улику теряется, и это стоит видеть отдельной строкой.
        low_confidence_evidence=sum(
            1
            for cell in answered
            if cell.outcome.status == "BREACH" and cell.outcome.evidence_txn_id is None
        ),
        llm_calls=sum(stage.calls for stage in stages.values()),
        cache_hits=sum(stage.cache_hits for stage in stages.values()),
        live_calls=sum(stage.live_calls for stage in stages.values()),
        tokens_in=sum(stage.tokens_in for stage in stages.values()),
        tokens_out=sum(stage.tokens_out for stage in stages.values()),
        estimated_cost_usd=engines.budget.estimated_cost,
        runtime_seconds=round(seconds, 3),
    )


def _write_json(path: Path, payload: Any) -> None:
    # Перевод строки задан явно: на Windows `write_text` иначе пишет CRLF, и отпечаток
    # содержимого перестаёт совпадать с отпечатком файла на диске.
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def solve(
    context: RunContext,
    input_path: Path,
    output_path: Path,
    *,
    template_path: Path | None = None,
    engines: Engines | None = None,
) -> SolveResult:
    """Сквозной прогон. Бросает `PipelineError`, если ответ собрался не полностью."""
    started = time.perf_counter()
    _require_header(context)

    dataset_root = unpack(input_path, context.root)
    template_file = find_template(dataset_root, template_path)
    template = SubmissionTemplate.load(template_file)
    tools = engines or build_engines(context)

    manifest = context.build_manifest(
        input_path,
        replayed_from=context.run_id if context.mode is RunMode.REPLAY else None,
        template_path=template_file,
        prompts=prompt_digests(),
    )
    context.write_manifest(manifest)

    # Перепись и реестр — общий вход всех заёмщиков. Отказ здесь означает, что считать
    # нечего вовсе, и разбирать его по заёмщикам нечестно: ни один из них не начинался.
    try:
        inventory = build_inventory(
            dataset_root,
            template.scenarios,
            tools.ocr,
            max_concurrency=context.settings.max_concurrency,
        )
        facts = build_facts(inventory)
    except CacheMissError as exc:
        # Распознавание идёт до всякого расчёта, и промах кэша здесь — не свойство
        # датасета, а запрет платить за живой вызов. Причины у отказа разные, значит
        # и сообщения разные.
        context.finish(manifest, RunStatus.FAILED)
        raise PipelineError(f"Страница требует распознавания живым вызовом: {exc}") from exc
    except (InventoryError, LedgerError, ScenarioMappingError) as exc:
        context.finish(manifest, RunStatus.FAILED)
        raise PipelineError(f"Датасет не разобран: {exc}") from exc

    violations = [*check_documents(inventory), *check_facts(facts, inventory)]

    results = _run_borrowers(
        template,
        inventory,
        facts,
        tools,
        max_concurrency=context.settings.max_concurrency,
    )
    answers: dict[tuple[str, str], Outcome] = {
        cell.cell: cell.outcome for result in results for cell in result.answered
    }
    violations += check_coverage(
        [f"{scenario}/{clause}" for scenario, clause in answers],
        [f"{scenario}/{clause}" for scenario, clause in template.cells],
    )

    report = RunReport(
        run_id=context.run_id,
        mode=context.mode.value,
        borrowers=[_borrower_report(result) for result in results],
        invariants=summarise(violations),
        budget=tools.budget.report(),
        template_cells=template.cells,
        answered_cells=tuple(answers),
        failures=_errors(violations),
        notes=[*_warnings(violations), *_rejections(tools)],
        last_known_good=keeper.status(context.settings),
    )

    document = (
        build_submission(template, context.settings, answers).model_dump_json(indent=2)
        if report.is_clean
        else ""
    )
    submission_sha: str | None = None
    if document:
        # Файл ответа переписывается только успехом. Упавший прогон оставляет
        # предыдущий полный ответ нетронутым — подмешивать в него свои ячейки нельзя,
        # а забирать последнее, что было, тем более.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8", newline="\n")
        context.submission_path.write_text(document, encoding="utf-8", newline="\n")
        submission_sha = sha256_bytes(document.encode("utf-8"))
        report.last_known_good = keeper.remember(
            context.settings, run_id=context.run_id, output_path=output_path, sha256=submission_sha
        ).record() | {"state": "intact", "detail": f"{output_path} — ответ этого прогона"}

    _write_lineage(context, results)
    metrics = _metrics(
        inventory,
        template,
        results,
        engines=tools,
        violations=violations,
        seconds=time.perf_counter() - started,
    )
    _write_json(context.metrics_path, metrics.model_dump(mode="json"))
    _write_json(context.report_path, report.record())
    # Индекс — это записи кэша, которыми пользовался именно этот прогон. Отпечаток
    # всего общего каталога менялся бы от чужой работы и о нашей не говорил бы ничего.
    _write_json(context.cache_index_path, tools.journal.index())
    context.finish(
        manifest,
        RunStatus.COMPLETED if report.is_clean else RunStatus.FAILED,
        submission_sha256=submission_sha,
        model_cache_sha256=tools.journal.digest,
    )

    if not report.is_clean:
        raise PipelineError(
            f"Прогон не дал полного ответа: {report.summary()}. "
            f"Подробности в {context.report_path}. "
            f"Предыдущий ответ: {report.last_known_good.get('detail', '')}",
            report,
        )
    return SolveResult(
        submission_path=output_path, metrics=metrics, report=report, run_dir=context.root
    )
