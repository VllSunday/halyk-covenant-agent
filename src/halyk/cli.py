from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from halyk import __version__
from halyk.config import Settings
from halyk.models.manifest import RunMode
from halyk.output.explain import render_explanation
from halyk.output.validator import validate_file
from halyk.pipeline import solve as run_pipeline
from halyk.run.context import LINEAGE_NAME, RunContext
from halyk.run.trace import read_lineage

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Проверка ковенантов по кредитным документам и реестру транзакций.",
)
console = Console()
error_console = Console(stderr=True, style="red")


def _latest_run(artifacts_dir: Path) -> Path:
    runs = artifacts_dir / "runs"
    candidates = (
        sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True) if runs.exists() else []
    )
    if not candidates:
        error_console.print(f"В {runs} нет ни одного прогона")
        raise typer.Exit(code=1)
    return candidates[0]


@app.command()
def version() -> None:
    """Версия пакета."""
    console.print(__version__)


@app.command()
def solve(
    input_path: Annotated[Path, typer.Option("--input", exists=True, help="Архив с датасетом")],
    output_path: Annotated[Path, typer.Option("--output", help="Куда записать ответ")] = Path(
        "Submission.json"
    ),
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
        typer.Option("--cache-read", help="Разрешить чтение кэша при новом прогоне"),
    ] = False,
) -> None:
    """Полный прогон от архива до Submission.json.

    По умолчанию считает вживую и кэш только пишет. Чтение кэша включается явно —
    иначе новый датасет мог бы получить ответы от прошлого прогона.
    """
    if resume and replay:
        error_console.print("--resume и --replay вместе не имеют смысла")
        raise typer.Exit(code=2)

    settings = Settings.from_env()
    mode = RunMode.REPLAY if replay else RunMode.RESUME if resume else RunMode.SOLVE
    context = RunContext.create(
        settings=settings,
        input_path=input_path,
        mode=mode,
        run_id=resume or (replay.name if replay else None),
    )
    if cache_read and mode is RunMode.SOLVE:
        console.print("[yellow]Чтение кэша включено вручную[/yellow]")

    result = run_pipeline(context, input_path, output_path)
    console.print(f"Готово: {result.submission_path}")


@app.command()
def validate(
    submission: Annotated[Path, typer.Option("--submission", exists=True)] = Path(
        "Submission.json"
    ),
    schema_dir: Annotated[Path, typer.Option("--schema-dir")] = Path("schemas"),
) -> None:
    """Проверить ответ по JSON Schema."""
    issues = validate_file(submission, "submission", schema_dir)
    if not issues:
        console.print(f"[green]{submission} соответствует схеме[/green]")
        return

    for issue in issues:
        error_console.print(f"{issue.location}: {issue.message}")
    error_console.print(f"\nВсего нарушений: {len(issues)}")
    raise typer.Exit(code=1)


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
