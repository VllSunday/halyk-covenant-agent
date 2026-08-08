"""Офлайн-режим: живой вызов недостижим, промах кэша виден.

Проверка идёт не по флагу, а по факту: конструктор клиента подменяется взрывающимся,
и любое обращение к сети превратилось бы в падение теста.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from halyk.config import ConfigError, ModelConfig, OfflineError, Settings
from halyk.llm.cache import CacheMissError, CachePolicy, ModelCache
from halyk.llm.classify import CategoryClassifier, TransactionInput


class ClientReachedError(AssertionError):
    """Поднимается вместо создания клиента: в офлайне сюда попадать нельзя."""


@pytest.fixture
def exploding_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise ClientReachedError("создан сетевой клиент в офлайн-режиме")

    monkeypatch.setattr("openai.OpenAI", explode)


def offline_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = {"name": "test-model", "api_key": "sk-test", "offline": True}
    return ModelConfig(**(base | overrides))  # type: ignore[arg-type]


def test_authorise_refuses_before_any_client_is_built() -> None:
    with pytest.raises(OfflineError, match="Офлайн-режим"):
        offline_config().authorise_live_call("проверка")


def test_authorise_passes_the_key_when_online() -> None:
    config = offline_config(offline=False)
    assert config.authorise_live_call("проверка") == "sk-test"


def test_missing_key_is_still_a_config_error() -> None:
    config = offline_config(offline=False, api_key=None)
    with pytest.raises(ConfigError):
        config.authorise_live_call("проверка")


def test_offline_wins_over_a_missing_key() -> None:
    """Порядок проверок важен: в офлайне отсутствие ключа не должно маскировать причину."""
    with pytest.raises(OfflineError):
        offline_config(api_key=None).authorise_live_call("проверка")


def test_cache_miss_names_the_offline_mode(tmp_path: Path) -> None:
    cache = ModelCache(directory=tmp_path, policy=CachePolicy.OFFLINE)
    with pytest.raises(CacheMissError, match="CACHE_MISS"):
        cache.get("ключа-нет")


def test_cache_hit_works_offline(tmp_path: Path) -> None:
    writable = ModelCache(directory=tmp_path, policy=CachePolicy.READ_WRITE)
    writable.put("ключ", {"items": []})

    offline = ModelCache(directory=tmp_path, policy=CachePolicy.OFFLINE)
    assert offline.get("ключ") == {"items": []}


def test_classifier_never_reaches_the_network(tmp_path: Path, exploding_client: None) -> None:
    """Сквозная проверка: промах кэша обрывается до создания клиента."""
    classifier = CategoryClassifier(
        config=offline_config(),
        cache=ModelCache(directory=tmp_path, policy=CachePolicy.OFFLINE),
    )
    rows = [
        TransactionInput(
            txn_id="TXN-P1-0001",
            description="Оплата электроэнергии",
            counterparty="Энергосбыт",
            amount="-1000.00",
            currency="USD",
            date="2025-03-01",
        )
    ]
    with pytest.raises(CacheMissError):
        classifier.classify("ACC-0001", rows)
    assert classifier.live_calls == 0


def test_settings_read_the_offline_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HALYK_OFFLINE", "1")
    settings = Settings.from_env()
    assert settings.offline
    assert all(
        config.offline
        for config in (settings.compiler, settings.ocr, settings.classifier, settings.verifier)
    )


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_offline_variable_stays_off_for_falsy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HALYK_OFFLINE", value)
    assert not Settings.from_env().offline


def test_flag_turns_offline_on_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HALYK_OFFLINE", raising=False)
    assert Settings.from_env(offline=True).offline


def test_settings_separate_per_call_and_total_token_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HALYK_MAX_INPUT_TOKENS_PER_CALL", "60000")
    monkeypatch.setenv("HALYK_MAX_TOTAL_INPUT_TOKENS", "100000")
    monkeypatch.setenv("HALYK_MAX_OUTPUT_TOKENS", "16000")

    settings = Settings.from_env()

    assert settings.max_input_tokens_per_call == 60000
    assert settings.max_total_input_tokens == 100000
    assert settings.max_output_tokens == 16000
    assert settings.compiler.max_output_tokens == 16000


def test_old_input_limit_name_remains_a_total_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HALYK_MAX_TOTAL_INPUT_TOKENS", raising=False)
    monkeypatch.setenv("HALYK_MAX_INPUT_TOKENS", "90000")

    assert Settings.from_env().max_total_input_tokens == 90000
