"""Проверка доступа к модели не должна стоить дороже самой проверки."""

from typing import Any

import openai
import pytest
from typer.testing import CliRunner

from halyk.cli import app


class FakeResponses:
    def __init__(self, log: list[dict[str, Any]]) -> None:
        self.log = log

    def create(self, **request: Any) -> Any:
        self.log.append(request)
        return type("Response", (), {"usage": type("Usage", (), {"total_tokens": 12})()})()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            self.responses = FakeResponses(log)

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Уровни задаются явно: иначе тест читал бы .env разработчика и проверял бы его
    # настройки вместо того, какая роль ушла в какой вызов.
    monkeypatch.setenv("HALYK_OCR_EFFORT", "low")
    monkeypatch.setenv("HALYK_COMPILER_EFFORT", "high")
    monkeypatch.setenv("HALYK_CLASSIFIER_EFFORT", "medium")
    monkeypatch.setenv("HALYK_VERIFIER_EFFORT", "xhigh")
    return log


def test_single_role_makes_a_single_call(calls: list[dict[str, Any]]) -> None:
    """Перед распознаванием нужен доступ к OCR, а не прогрев верификатора на xhigh."""
    result = CliRunner().invoke(app, ["smoke", "--role", "ocr"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["reasoning"]["effort"] == "low"


def test_all_roles_are_checked_by_default(calls: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(app, ["smoke"])
    assert result.exit_code == 0, result.output
    assert [call["reasoning"]["effort"] for call in calls] == ["low", "high", "medium", "xhigh"]


def test_unknown_role_is_rejected_before_any_call(calls: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(app, ["smoke", "--role", "compilator"])
    assert result.exit_code == 2
    assert calls == []
