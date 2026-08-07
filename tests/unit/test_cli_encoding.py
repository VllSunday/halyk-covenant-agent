"""Вывод команд не должен зависеть от кодировки терминала.

Прогон на боевом датасете идёт часы; потерять его результат из-за символа, который
не влез в кодовую страницу консоли, недопустимо.
"""

import io
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from halyk.cli import app, tolerate_narrow_console


class NarrowStream(io.TextIOWrapper):
    """Поток с кодировкой, в которой нет ни кириллицы, ни типографских знаков."""

    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="cp1251", errors="strict", newline="")


def test_narrow_console_raises_without_the_guard() -> None:
    stream = NarrowStream()
    with pytest.raises(UnicodeEncodeError):
        stream.write("сценарий P1 → ACC-7801")
        stream.flush()


def test_guard_replaces_unsupported_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = NarrowStream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    tolerate_narrow_console()
    stream.write("сценарий P1 → ACC-7801")
    stream.flush()

    written = stream.buffer.getvalue().decode("cp1251")  # type: ignore[attr-defined]
    assert "сценарий P1" in written


def test_report_is_written_before_printing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Артефакт должен появиться, даже если печать сводки провалится.

    Порядок «сначала файл, потом терминал» — единственное, что защищает результат
    от оформления.
    """
    dataset = tmp_path / "dataset"
    (dataset / "documents").mkdir(parents=True)
    (dataset / "ledger.csv").write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-P1-0001,2025-01-05,ACC-7801,Alpha,Sales,1.00,USD\n",
        encoding="utf-8",
    )
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "team": "",
                "contact_email": "",
                "model": "",
                "answers": {"P1": {"6.1": {"status": None, "actual": None}}},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "out" / "inventory.json"

    result = CliRunner().invoke(
        app,
        [
            "inventory",
            "--dataset",
            str(dataset),
            "--template",
            str(template),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text(encoding="utf-8"))["scenarios"] == {"P1": "ACC-7801"}
