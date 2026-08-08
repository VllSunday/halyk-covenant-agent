"""Общий кэш и последний полный ответ.

Два свойства, которые дороже всего проверять на боевых деньгах. Первое: работа,
оплаченная переписью, не оплачивается прогоном второй раз. Второе: неудачная попытка
не уносит с собой ответ, который уже был получен.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from halyk.pipeline import Engines, PipelineError, solve
from halyk.pipeline.keeper import marker_path
from halyk.run.context import RunContext

from .conftest import E1_ACCOUNT, E2_ACCOUNT, compiled, e1_clauses, e2_clauses, prime_ocr_cache


def test_page_recognised_by_the_inventory_is_a_cache_hit_in_the_run(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """`halyk inventory --ocr` и `solve` спрашивают одно и то же и платят один раз."""
    context = make_context()
    primed = prime_ocr_cache(context.settings, dataset)
    assert primed.calls == 1

    engines = make_engines(context)
    solve(context, dataset, tmp_path / "Submission.json", engines=engines)

    assert engines.ocr is not None
    assert engines.ocr.live_calls == 0
    assert engines.ocr.cache_hits == 1
    assert engines.ocr.engine.calls == 0  # type: ignore[attr-defined]


def test_run_records_only_the_cache_entries_it_used(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Индекс — про этот прогон, а не про весь каталог кэша."""
    context = make_context()
    engines = make_engines(context)
    solve(context, dataset, tmp_path / "Submission.json", engines=engines)

    index = json.loads(context.cache_index_path.read_text(encoding="utf-8"))
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    roles = {entry["role"] for entry in index["entries"]}

    assert manifest["model_cache_sha256"] == index["digest"]
    assert roles == {"ocr", "compiler", "resolver", "categories", "categories-verifier"}
    assert index["written"] == len(index["entries"])
    assert index["reused"] == 0
    assert all(len(entry["sha256"]) == 64 for entry in index["entries"])


def test_second_run_reuses_the_shared_cache_without_resume(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Кэш общий, поэтому переиспользование не требует того же run_id."""
    first = make_context(run_id="first")
    solve(first, dataset, tmp_path / "First.json", engines=make_engines(first))

    second = make_context(run_id="second")
    engines = make_engines(second)
    solve(second, dataset, tmp_path / "Second.json", engines=engines)

    index = json.loads(second.cache_index_path.read_text(encoding="utf-8"))
    assert index["written"] == 0
    assert index["reused"] == len(index["entries"])
    assert engines.compiler.runner.send.sent == []  # type: ignore[attr-defined]
    assert (tmp_path / "Second.json").read_text(encoding="utf-8") == (
        tmp_path / "First.json"
    ).read_text(encoding="utf-8")


def test_no_cache_read_asks_the_models_again(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Отказ от чтения кэша — осознанное решение переспросить, и оно работает."""
    first = make_context(run_id="first")
    solve(first, dataset, tmp_path / "First.json", engines=make_engines(first))

    second = make_context(run_id="second", fresh=True)
    engines = make_engines(second)
    solve(second, dataset, tmp_path / "Second.json", engines=engines)

    assert engines.compiler.runner.send.accounts == [E1_ACCOUNT, E2_ACCOUNT]  # type: ignore[attr-defined]
    index = json.loads(second.cache_index_path.read_text(encoding="utf-8"))
    assert index["reused"] == 0


def test_failed_run_leaves_the_previous_answer_in_place(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Упавшая попытка не уносит ответ, который уже был получен."""
    output = tmp_path / "Submission.json"
    good = make_context(run_id="good")
    solve(good, dataset, output, engines=make_engines(good))
    kept = output.read_text(encoding="utf-8")

    # Переспрашиваем модель заново: иначе прогон прочитал бы удачный ответ из общего
    # кэша и упасть ему было бы не на чем.
    broken = make_context(run_id="broken", fresh=True)
    answers = {E1_ACCOUNT: [compiled(e1_clauses()[:1])], E2_ACCOUNT: [compiled(e2_clauses())]}
    with pytest.raises(PipelineError, match="Предыдущий ответ"):
        solve(broken, dataset, output, engines=make_engines(broken, compiler=answers))

    assert output.read_text(encoding="utf-8") == kept
    report = json.loads(broken.report_path.read_text(encoding="utf-8"))
    assert report["last_known_good"]["state"] == "intact"
    assert report["last_known_good"]["run_id"] == "good"
    assert not broken.submission_path.exists()


def test_marker_notices_a_hand_edited_answer(
    dataset: Path,
    tmp_path: Path,
    make_context: Callable[..., RunContext],
    make_engines: Callable[..., Engines],
) -> None:
    """Правку файла руками видно по отпечатку, а по дате — нет."""
    output = tmp_path / "Submission.json"
    good = make_context(run_id="good")
    solve(good, dataset, output, engines=make_engines(good))
    assert marker_path(good.settings).exists()
    output.write_text("{}", encoding="utf-8")

    # Переспрашиваем модель заново: иначе прогон прочитал бы удачный ответ из общего
    # кэша и упасть ему было бы не на чем.
    broken = make_context(run_id="broken", fresh=True)
    answers = {E1_ACCOUNT: [compiled(e1_clauses()[:1])], E2_ACCOUNT: [compiled(e2_clauses())]}
    with pytest.raises(PipelineError):
        solve(broken, dataset, output, engines=make_engines(broken, compiler=answers))

    report = json.loads(broken.report_path.read_text(encoding="utf-8"))
    assert report["last_known_good"]["state"] == "modified"
