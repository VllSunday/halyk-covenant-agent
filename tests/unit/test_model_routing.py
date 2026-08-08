"""Роли получают модель по сложности, а манифест показывает весь каскад."""

from decimal import Decimal

import pytest

from halyk.config import Settings


@pytest.fixture(autouse=True)
def routing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HALYK_OCR_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("HALYK_CLASSIFIER_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("HALYK_VERIFIER_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("HALYK_COMPILER_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("HALYK_RESOLVER_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("HALYK_FALLBACK_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("HALYK_OCR_EFFORT", "low")
    monkeypatch.setenv("HALYK_CLASSIFIER_EFFORT", "medium")
    monkeypatch.setenv("HALYK_VERIFIER_EFFORT", "medium")
    monkeypatch.setenv("HALYK_COMPILER_EFFORT", "high")
    monkeypatch.setenv("HALYK_RESOLVER_EFFORT", "high")
    monkeypatch.setenv("HALYK_FALLBACK_EFFORT", "high")


def test_role_routing_and_prices() -> None:
    settings = Settings.from_env()

    assert (settings.ocr.name, settings.ocr.reasoning_effort) == ("gpt-5.6-terra", "low")
    assert (settings.classifier.name, settings.classifier.reasoning_effort) == (
        "gpt-5.6-luna",
        "medium",
    )
    assert (settings.verifier.name, settings.verifier.reasoning_effort) == (
        "gpt-5.6-terra",
        "medium",
    )
    assert (settings.compiler.name, settings.compiler.reasoning_effort) == (
        "gpt-5.6-terra",
        "high",
    )
    assert (settings.resolver.name, settings.resolver.reasoning_effort) == (
        "gpt-5.6-terra",
        "high",
    )
    assert (settings.fallback.name, settings.fallback.reasoning_effort) == (
        "gpt-5.6-sol",
        "high",
    )
    assert settings.classifier.price_input_per_million == Decimal("0.2")
    assert settings.compiler.price_input_per_million == Decimal("2")
    assert settings.fallback.price_output_per_million == Decimal("30")


def test_manifest_names_every_model_role() -> None:
    assert Settings.from_env().model_versions() == {
        "covenant_compiler": "gpt-5.6-terra",
        "fact_resolver": "gpt-5.6-terra",
        "failure_escalation": "gpt-5.6-sol",
        "ocr": "gpt-5.6-terra",
        "classifier": "gpt-5.6-luna",
        "verifier": "gpt-5.6-terra",
    }
