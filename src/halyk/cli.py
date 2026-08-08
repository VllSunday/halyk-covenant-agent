from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from halyk import __version__
from halyk.config import ConfigError, OfflineError, Settings
from halyk.models.manifest import RunMode
from halyk.models.submission import Submission
from halyk.output.explain import render_explanation
from halyk.output.template import SubmissionTemplate
from halyk.output.validator import validate_file
from halyk.pipeline import PipelineError
from halyk.pipeline import solve as run_pipeline
from halyk.pipeline.dataset import DatasetError
from halyk.pipeline.report import first_lines
from halyk.run.context import LINEAGE_NAME, RunContext
from halyk.run.trace import read_lineage


def tolerate_narrow_console() -> None:
    """Не падать на символе, которого нет в кодировке терминала.

    Консоль Windows по умолчанию не UTF-8, и один такой символ обрывает команду
    исключением уже после того, как вся работа сделана. Замена на вопросительный
    знак — единственное поведение, при котором результат прогона не теряется из-за
    оформления.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


tolerate_narrow_console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Проверка ковенантов по кредитным документам и реестру транзакций.",
)
console = Console()
error_console = Console(stderr=True, style="red")

SMOKE_ROLES = ("ocr", "compiler", "classifier", "verifier")


def _latest_run(artifacts_dir: Path) -> Path:
    runs = artifacts_dir / "runs"
    candidates = (
        sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True) if runs.exists() else []
    )
    if not candidates:
        error_console.print(f"В {runs} нет ни одного прогона")
        raise typer.Exit(code=1)
    return candidates[0]


def _ocr_engine(settings: Settings, *, enabled: bool) -> Any:
    """Движок распознавания для команд переписи. Ключ проверяется до разбора файлов."""
    from halyk.llm.cache import CachePolicy, ModelCache  # noqa: PLC0415
    from halyk.parsing.ocr import CachedOcr, OpenAIVisionOcr  # noqa: PLC0415

    if not enabled:
        return None
    if not settings.offline:
        try:
            settings.ocr.authorise_live_call("проверка доступа перед распознаванием")
        except ConfigError as exc:
            error_console.print(str(exc))
            raise typer.Exit(code=2) from exc
    return CachedOcr(
        engine=OpenAIVisionOcr(config=settings.ocr),
        cache=ModelCache(
            directory=settings.artifacts_dir / "cache" / "ocr",
            policy=CachePolicy.OFFLINE if settings.offline else CachePolicy.READ_WRITE,
        ),
    )


@app.command()
def version() -> None:
    """Версия пакета."""
    console.print(__version__)


@app.command()
def solve(
    input_path: Annotated[Path, typer.Option("--input", exists=True, help="Архив с датасетом")],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Куда записать ответ; файл переписывается прогоном"),
    ] = Path("Submission.json"),
    *,
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            exists=True,
            help="Шаблон ответа; по умолчанию берётся из самого датасета",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Продолжить прогон с указанным run_id, используя его кэш"),
    ] = None,
    replay: Annotated[
        Path | None,
        typer.Option("--replay", help="Повторить прогон из каталога artifacts/runs/<run_id>"),
    ] = None,
    cache_read: Annotated[
        bool,
        typer.Option(
            "--cache-read",
            help="Разрешить чтение кэша прогона; кэш лежит в его каталоге, см. --resume",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            "--cache-only",
            help="Считать только по кэшу: сетевой клиент не создаётся, промах — ошибка",
        ),
    ] = False,
) -> None:
    """Полный прогон от архива до Submission.json.

    По умолчанию считает вживую и кэш только пишет. Чтение кэша включается явно —
    иначе новый датасет мог бы получить ответы от прошлого прогона.

    Прогон строгий: незакрытая ячейка заканчивается ненулевым кодом возврата и
    отчётом, а не файлом ответа с пропусками.
    """
    if resume and replay:
        error_console.print("--resume и --replay вместе не имеют смысла")
        raise typer.Exit(code=2)

    settings = Settings.from_env(offline=offline)
    mode = RunMode.REPLAY if replay else RunMode.RESUME if resume else RunMode.SOLVE
    context = RunContext.create(
        settings=settings,
        input_path=input_path,
        mode=mode,
        run_id=resume or (replay.name if replay else None),
        cache_read=cache_read,
    )
    if cache_read and mode is RunMode.SOLVE:
        console.print("[yellow]Чтение кэша включено вручную[/yellow]")

    try:
        result = run_pipeline(context, input_path, output_path, template_path=template)
    except DatasetError as exc:
        error_console.print(str(exc))
        raise typer.Exit(code=2) from exc
    except PipelineError as exc:
        error_console.print(str(exc))
        if exc.report is not None:
            for line in first_lines(exc.report.problems):
                error_console.print(f"  {line}")
        raise typer.Exit(code=1) from exc

    metrics = result.metrics
    console.print(f"Готово: {result.submission_path}")
    console.print(
        f"Ячеек {metrics.covenants_answered} из {metrics.covenants_found}, "
        f"заёмщиков {metrics.borrowers_total}, операций {metrics.transactions_total}"
    )
    console.print(
        f"Вызовов моделей {metrics.llm_calls} (живых {metrics.live_calls}, "
        f"из кэша {metrics.cache_hits}), {metrics.runtime_seconds:.1f} c"
    )
    console.print(f"Артефакты прогона: {result.run_dir}")
    for note in result.report.notes:
        console.print(f"[yellow]{note.code}[/yellow] {note.subject}: {note.detail}")


@app.command()
def validate(
    submission: Annotated[Path, typer.Option("--submission", exists=True)] = Path(
        "Submission.json"
    ),
    schema_dir: Annotated[Path, typer.Option("--schema-dir")] = Path("schemas"),
    template: Annotated[
        Path | None,
        typer.Option("--template", exists=True, help="Шаблон организаторов для сверки ячеек"),
    ] = None,
) -> None:
    """Проверить ответ по JSON Schema, а с --template ещё и по составу ячеек."""
    issues = validate_file(submission, "submission", schema_dir)

    # Состав ячеек сверяем только у документа, прошедшего схему: разбирать в модель
    # заведомо поломанный JSON смысла нет, а сообщение об ошибке выйдет хуже.
    if template is not None and not issues:
        parsed = Submission.model_validate_json(submission.read_text(encoding="utf-8"))
        issues.extend(SubmissionTemplate.load(template).check(parsed))

    if not issues:
        checked = "схеме и шаблону" if template else "схеме"
        console.print(f"[green]{submission} соответствует {checked}[/green]")
        return

    for issue in issues:
        error_console.print(f"{issue.location}: {issue.message}")
    error_console.print(f"\nВсего нарушений: {len(issues)}")
    raise typer.Exit(code=1)


@app.command()
def score(
    submission: Annotated[Path, typer.Option("--submission", exists=True)] = Path(
        "Submission.json"
    ),
    ground_truth: Annotated[Path, typer.Option("--ground-truth", exists=True)] = Path(
        "agentic-bank-public/ground_truth.json"
    ),
    show_all: Annotated[
        bool, typer.Option("--all", help="Показать все ячейки, а не только потерявшие баллы")
    ] = False,
) -> None:
    """Оценить ответ по ключу открытого датасета.

    Команда для разработки. Ключ есть только у открытого набора, и расчётный путь
    его не читает — иначе результат на публичных данных перестал бы что-либо значить.
    """
    # Импорт внутри команды, а не наверху модуля: так пакет halyk.dev не оказывается
    # в дереве импортов solve, и запрет проверяется тестом, а не на честном слове.
    from halyk.dev.scoring import load_ground_truth, score_submission  # noqa: PLC0415

    parsed = Submission.model_validate_json(submission.read_text(encoding="utf-8"))
    report = score_submission(parsed, load_ground_truth(ground_truth))

    table = Table(title=f"{submission.name} против {ground_truth.name}")
    for column in ("ячейка", "вердикт", "число", "улика", "балл", "что не так"):
        table.add_column(column)
    for cell in report.cells:
        if not show_all and cell.total == 1:
            continue
        error = cell.relative_error
        table.add_row(
            f"{cell.scenario}/{cell.covenant}",
            f"{cell.status_points:.2f}",
            f"{cell.actual_points:.2f}" + (f" (±{error:.2%})" if error else ""),
            f"{cell.evidence_points:.2f}",
            f"{cell.total:.2f}",
            cell.note,
        )
    if table.row_count:
        console.print(table)

    parts = ", ".join(f"{name} {value:.2f}" for name, value in report.components.items())
    console.print(
        f"Итого [bold]{report.total:.2f}[/bold] из {report.max_total}, "
        f"без потерь {report.exact_cells} из {len(report.cells)}, {parts}"
    )
    console.print("[dim]Веса ячеек по сложности организаторы не раскрыли, итог невзвешенный[/dim]")

    # Расхождение состава ячеек делает итог несопоставимым, поэтому он не молчит и
    # не остаётся «зелёным»: ответ с посторонней ячейкой прошёл бы на полный балл.
    if not report.is_comparable:
        for scenario, covenant in report.missing:
            error_console.print(f"answers/{scenario}/{covenant}: ячейки нет в ответе")
        for scenario, covenant in report.unexpected:
            error_console.print(f"answers/{scenario}/{covenant}: ячейки нет в ключе")
        error_console.print("\nСостав ячеек разошёлся с ключом, итог сравнивать нельзя")
        raise typer.Exit(code=1)


@app.command()
def inventory(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    template: Annotated[Path, typer.Option("--template", exists=True)],
    ocr: Annotated[
        bool, typer.Option("--ocr", help="Распознавать страницы без пригодного текста")
    ] = False,
    report_path: Annotated[
        Path | None, typer.Option("--report", help="Куда записать сводку в JSON")
    ] = None,
) -> None:
    """Перепись датасета: типы документов, привязка к счетам, план распознавания."""
    from halyk.ingest.inventory import build_inventory  # noqa: PLC0415
    from halyk.output.template import SubmissionTemplate  # noqa: PLC0415

    settings = Settings.from_env()
    scenarios = SubmissionTemplate.load(template).scenarios
    engine = _ocr_engine(settings, enabled=ocr)
    found = build_inventory(dataset, scenarios, engine)
    summary = found.report()

    # Артефакт пишется до вывода в терминал: печать зависит от кодировки консоли и
    # может упасть, а результат многочасового прогона терять из-за этого нельзя.
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    kinds = Table(title="Документы по типам")
    kinds.add_column("тип")
    kinds.add_column("штук", justify="right")
    for kind, count in summary["kinds"].items():
        kinds.add_row(kind, str(count))
    console.print(kinds)

    console.print(
        f"Файлов {summary['files']}, PDF {summary['pdf']}, страниц {summary['pages']}, "
        f"реестр {found.ledger_path.name} ({len(found.ledger)} строк)"
    )
    console.print(
        f"Сценариев {len(summary['scenarios'])}: "
        + ", ".join(f"{s} -> {a}" for s, a in summary["scenarios"].items())
    )
    if engine is not None:
        usage = summary["ocr_usage"]
        console.print(
            f"Вызовов OCR: живых {usage['live_calls']}, из кэша {usage['cache_hits']}, "
            f"токенов {usage['total_tokens']}, {usage['seconds_total']:.1f} c "
            f"(дольше всех {usage['seconds_slowest']:.1f} c)"
        )

    if summary["ocr_pages"]:
        pages = Table(title="Страницы без пригодного текстового слоя")
        for column in ("документ", "стр.", "причина"):
            pages.add_column(column)
        for item in summary["ocr_pages"]:
            pages.add_row(item["document"], str(item["page"]), item["reason"])
        console.print(pages)
        if not ocr:
            console.print("[yellow]Запущено без --ocr: текст этих страниц не извлечён[/yellow]")

    console.print(
        f"Без счёта среди классифицированных: {len(summary['unresolved_documents'])}, "
        f"ждут распознавания: {len(summary['unclassified_pending_ocr'])}"
    )
    if summary["unresolved_documents"]:
        error_console.print(
            "Значимые документы без счёта: " + ", ".join(summary["unresolved_documents"])
        )

    if report_path is not None:
        console.print(f"Сводка: {report_path}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _fact_artifacts(out_dir: Path, collected: Any, ledger: Any) -> dict[str, Any]:
    """Выгрузка слоя фактов. Пишется до вывода в терминал, как и сводка переписи."""
    from halyk.hashing import sha256_payload  # noqa: PLC0415

    records = {
        "facts": [fact.record() for fact in collected.facts],
        "adjustments": [item.record() for item in ledger.adjustments],
        "transactions": [txn.record() for txn in ledger.transactions],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in records.items():
        _write_jsonl(out_dir / f"{name}.jsonl", rows)

    summary = {
        "period": ledger.period.record(),
        "facts": len(records["facts"]),
        "adjustments": len(records["adjustments"]),
        "applied": len(ledger.applied),
        "problems": [
            {"id": item.id, "status": item.status.value, "selector": item.selector.describe()}
            for item in ledger.problems
        ],
        "unparsed_dossiers": list(collected.unparsed_dossiers),
        "transactions": len(records["transactions"]),
        "ledger_digest": sha256_payload(records["transactions"]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


@app.command()
def facts(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    template: Annotated[Path, typer.Option("--template", exists=True)],
    *,
    ocr: Annotated[
        bool, typer.Option("--ocr", help="Распознавать страницы без пригодного текста")
    ] = False,
    out_dir: Annotated[Path, typer.Option("--out", help="Каталог для JSONL-артефактов")] = Path(
        "artifacts/facts"
    ),
    # Значения открытого набора — baseline для разработки. В сквозном прогоне период
    # обязан приходить из IR ковенанта, см. docs/adr/0008.
    period_start: Annotated[
        str, typer.Option("--period-start", help="Начало ковенантного периода, ГГГГ-ММ-ДД")
    ] = "2025-01-01",
    period_end: Annotated[
        str, typer.Option("--period-end", help="Конец ковенантного периода, ГГГГ-ММ-ДД")
    ] = "2025-12-31",
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Ненулевой код возврата при неучтённых документах"),
    ] = False,
) -> None:
    """Извлечь факты из документов и применить корректировки к реестру.

    Ковенантный период задаётся флагами и попадает в сводку: от него зависит,
    какие операции считаются относящимися к отчётному году, и умалчивать о нём
    в артефакте нельзя.
    """
    from halyk.ingest.inventory import build_inventory  # noqa: PLC0415
    from halyk.ingest.normalise import CovenantPeriod, normalise  # noqa: PLC0415
    from halyk.knowledge.facts import build_facts  # noqa: PLC0415
    from halyk.output.template import SubmissionTemplate  # noqa: PLC0415

    settings = Settings.from_env()
    engine = _ocr_engine(settings, enabled=ocr)
    try:
        period = CovenantPeriod(
            start=date.fromisoformat(period_start), end=date.fromisoformat(period_end)
        )
    except ValueError as exc:
        error_console.print(f"Границы периода задаются как ГГГГ-ММ-ДД: {exc}")
        raise typer.Exit(code=2) from exc

    inventory = build_inventory(dataset, SubmissionTemplate.load(template).scenarios, engine)
    collected = build_facts(inventory)
    ledger = normalise(inventory.ledger, collected.adjustments, period)
    summary = _fact_artifacts(out_dir, collected, ledger)

    kinds = Table(title="Факты по типам")
    kinds.add_column("тип")
    kinds.add_column("штук", justify="right")
    for kind, count in sorted(Counter(f.kind.value for f in collected.facts).items()):
        kinds.add_row(kind, str(count))
    console.print(kinds)

    statuses = Table(title="Корректировки по итогу")
    statuses.add_column("итог")
    statuses.add_column("штук", justify="right")
    for status, count in sorted(Counter(a.status.value for a in ledger.adjustments).items()):
        statuses.add_row(status, str(count))
    console.print(statuses)

    changed = sum(1 for txn in ledger.transactions if txn.is_adjusted)
    console.print(
        f"Период {period.start} - {period.end}, операций {summary['transactions']}, "
        f"изменено {changed}, отпечаток реестра {summary['ledger_digest'][:16]}"
    )
    console.print(f"Артефакты: {out_dir}")

    for dossier in collected.unparsed_dossiers:
        error_console.print(f"Досье без разобранного порога: {dossier}")
    for item in ledger.problems:
        error_console.print(
            f"{item.status.value}: {item.action.value} по {item.selector.describe()} "
            f"из {item.source.file_name} стр.{item.source.page}"
        )

    # Строгий режим для боевого прогона: неразобранное досье и неучтённый вывод
    # документа означают, что часть входных данных молча не дошла до расчёта.
    if strict and (ledger.problems or collected.unparsed_dossiers):
        error_console.print(
            f"\nНеучтённых документов: {len(ledger.problems) + len(collected.unparsed_dossiers)}"
        )
        raise typer.Exit(code=1)


@app.command()
def classify(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    template: Annotated[Path, typer.Option("--template", exists=True)],
    *,
    ocr: Annotated[
        bool, typer.Option("--ocr", help="Распознавать страницы без пригодного текста")
    ] = False,
    out_dir: Annotated[Path, typer.Option("--out", help="Каталог для артефактов")] = Path(
        "artifacts/classification"
    ),
    period_start: Annotated[str, typer.Option("--period-start")] = "2025-01-01",
    period_end: Annotated[str, typer.Option("--period-end")] = "2025-12-31",
    strict: Annotated[
        bool, typer.Option("--strict", help="Ненулевой код возврата при нераспознанных статьях")
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            "--cache-only",
            help="Считать только по кэшу: сетевой клиент не создаётся, промах — ошибка",
        ),
    ] = False,
) -> None:
    """Отнести операции заёмщиков к статьям, которыми оперируют ковенанты.

    Строгий режим — для разработки: он падает на первой же неразобранной операции.
    Боевой прогон так делать не должен, там одна неясная строка не может стоить
    зачёта по остальным ковенантам.
    """
    from halyk.ingest.inventory import build_inventory  # noqa: PLC0415
    from halyk.ingest.normalise import CovenantPeriod, normalise  # noqa: PLC0415
    from halyk.knowledge.classifier import TransactionClassifier  # noqa: PLC0415
    from halyk.knowledge.facts import build_facts  # noqa: PLC0415
    from halyk.llm.cache import CachePolicy, ModelCache  # noqa: PLC0415
    from halyk.llm.classify import CategoryClassifier, cache_signature  # noqa: PLC0415
    from halyk.output.template import SubmissionTemplate  # noqa: PLC0415

    settings = Settings.from_env(offline=offline)
    if not settings.offline:
        try:
            settings.classifier.authorise_live_call("проверка доступа перед классификацией")
        except ConfigError as exc:
            error_console.print(str(exc))
            raise typer.Exit(code=2) from exc

    engine = _ocr_engine(settings, enabled=ocr)
    period = CovenantPeriod(
        start=date.fromisoformat(period_start), end=date.fromisoformat(period_end)
    )
    scenarios = SubmissionTemplate.load(template).scenarios
    inventory = build_inventory(dataset, scenarios, engine)
    ledger = normalise(inventory.ledger, build_facts(inventory).adjustments, period)
    accounts = sorted(inventory.scenarios.scenario_to_account.values())

    def cache(name: str) -> ModelCache:
        return ModelCache(
            directory=settings.artifacts_dir / "cache" / name,
            policy=CachePolicy.OFFLINE if settings.offline else CachePolicy.READ_WRITE,
        )

    classifier = TransactionClassifier(
        model=CategoryClassifier(config=settings.classifier, cache=cache("categories")),
        verifier=CategoryClassifier(config=settings.verifier, cache=cache("categories-verifier")),
    )
    result = classifier.run(ledger, accounts)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "classification.jsonl", [r.record() for r in result.records])
    summary = {
        "prompt": cache_signature(),
        "matrix": result.matrix(),
        "usage": result.usage,
        "totals": result.totals(),
        "unresolved": [r.txn_id for r in result.unresolved],
        "conflicts": [r.txn_id for r in result.conflicts],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    table = Table(title="Как решено")
    table.add_column("основание")
    table.add_column("операций", justify="right")
    for source, count in result.matrix().items():
        table.add_row(source, str(count))
    console.print(table)

    usage = result.usage["model"]
    console.print(
        f"Батчей {usage['batches']}, живых {usage['live_calls']}, из кэша {usage['cache_hits']}, "
        f"токенов {usage['total_tokens']}, {usage['seconds_total']:.1f} c"
    )
    verifier = result.usage["verifier"]
    disputed = result.usage["verifier_disputed_rows"]
    if verifier:
        console.print(
            f"Verifier: спорных строк {disputed} в {verifier['batches']} батчах, "
            f"живых {verifier['live_calls']}, из кэша {verifier['cache_hits']}"
        )
    console.print(f"Артефакты: {out_dir}")

    for record in result.unresolved:
        error_console.print(f"без статьи: {record.txn_id} — {record.note}")
    if strict and result.unresolved:
        raise typer.Exit(code=1)


@app.command()
def smoke(
    role: Annotated[
        str | None,
        typer.Option("--role", help=f"Проверить только одну роль: {', '.join(SMOKE_ROLES)}"),
    ] = None,
) -> None:
    """Проверить доступ к модели и измерить задержку одного вызова.

    Наличие средств на счёте не означает доступа к конкретной модели, а узнавать об
    этом в боевом окне поздно. Заодно даёт первую точку по времени ответа.

    Роль выбирается, когда проверяется доступ перед распознаванием: верификатор
    работает на xhigh, и ждать его самый долгий вызов ради проверки ключа незачем.
    """
    import time  # noqa: PLC0415

    from openai import OpenAI  # noqa: PLC0415

    if role is not None and role not in SMOKE_ROLES:
        error_console.print(f"Неизвестная роль {role}. Допустимые: {', '.join(SMOKE_ROLES)}")
        raise typer.Exit(code=2)

    settings = Settings.from_env()
    by_role = {
        "ocr": settings.ocr,
        "compiler": settings.compiler,
        "classifier": settings.classifier,
        "verifier": settings.verifier,
    }
    for role_name, config in ((r, by_role[r]) for r in SMOKE_ROLES if role in (None, r)):
        # Отсутствие ключа — обычная ситуация настройки, а не сбой: показываем
        # причину строкой, без трассировки на пол-экрана.
        try:
            api_key = config.authorise_live_call(f"smoke-проверка роли {role_name}")
        except (ConfigError, OfflineError) as exc:
            error_console.print(str(exc))
            raise typer.Exit(code=2) from exc

        client = OpenAI(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        request: dict[str, Any] = {
            "model": config.name,
            "instructions": "Ответь одним словом.",
            "reasoning": {"effort": config.reasoning_effort},
            "input": "Скажи: готово",
        }
        started = time.perf_counter()
        try:
            response = client.responses.create(**request)
        except Exception as exc:
            error_console.print(f"{role_name} ({config.name}, {config.reasoning_effort}): {exc}")
            raise typer.Exit(code=1) from exc

        elapsed = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", "?") if usage else "?"
        console.print(
            f"[green]{role_name}[/green]: {config.name}, effort {config.reasoning_effort}, "
            f"{elapsed:.1f} c, токенов {tokens}"
        )


@app.command()
def explain(
    borrower_id: Annotated[str, typer.Argument(help="Идентификатор заёмщика")],
    covenant_id: Annotated[str, typer.Argument(help="Идентификатор ковенанта")],
    run: Annotated[
        Path | None,
        typer.Option("--run", help="Каталог прогона; по умолчанию последний"),
    ] = None,
) -> None:
    """Показать, откуда взялся конкретный ответ."""
    settings = Settings.from_env()
    run_dir = run or _latest_run(settings.artifacts_dir)

    records = read_lineage(run_dir / LINEAGE_NAME)
    matched = [r for r in records if r.borrower_id == borrower_id and r.covenant_id == covenant_id]
    if not matched:
        error_console.print(f"В прогоне {run_dir.name} нет ответа {borrower_id}/{covenant_id}")
        raise typer.Exit(code=1)

    for record in matched:
        console.print(render_explanation(record))
