"""Вызов модели: кэш, бюджет, повторы, телеметрия.

Сети здесь нет ни в одном тесте. Там, где нужен живой путь, подменяется отправка —
единственное место, где вызов уходит наружу.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.config import ModelConfig, OfflineError
from halyk.llm.cache import CacheMissError, CachePolicy, ModelCache
from halyk.llm.runner import (
    Budget,
    BudgetError,
    CascadingModelRunner,
    InvalidResponseError,
    Request,
    Role,
    StructuredModelRunner,
    batch_by_account,
    send_to_openai,
)
from halyk.pipeline.metrics import from_runner

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "integer"}}}


def request(payload: Any = None, *, account: str = "ACC-7801") -> Request:
    return Request(
        role=Role.COMPILER,
        account_id=account,
        instructions="переведи пункт в дерево",
        schema=SCHEMA,
        schema_name="covenant_formula",
        contract="covenant-formula-v2",
        payload=payload if payload is not None else {"clause": "6.1"},
    )


def parse(payload: dict[str, Any]) -> int:
    return int(payload["value"])


class Responder:
    """Подменяет единственное место, где запрос уходит наружу."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.sent = 0
        self.requests: list[Request] = []

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self.sent += 1
        candidate = args[1] if len(args) > 1 else None
        if isinstance(candidate, Request):
            self.requests.append(candidate)
        outcome = self.outcomes[min(self.sent - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, (100, 20), f"req-{self.sent}"


class ProviderError(RuntimeError):
    status_code = 400
    code = "invalid_json_schema"
    request_id = "req-rejected"


class QuotaError(RuntimeError):
    status_code = 429
    code = "insufficient_quota"
    request_id = "req-no-credit"


def runner(
    tmp_path: Path,
    *,
    policy: CachePolicy = CachePolicy.READ_WRITE,
    offline: bool = False,
    budget: Budget | None = None,
    transport_retries: int = 2,
    responder: Responder | None = None,
) -> StructuredModelRunner:
    return StructuredModelRunner(
        config=ModelConfig(name="test-model", api_key="sk-test", offline=offline),
        cache=ModelCache(directory=tmp_path, policy=policy),
        budget=budget or Budget(),
        transport_retries=transport_retries,
        send=responder or Responder({"value": 0}),
    )


# --- кэш ----------------------------------------------------------------------


def test_answer_is_cached_and_reused(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=(responder := Responder({"value": 7})))

    assert engine.run(request(), parse) == 7
    assert engine.run(request(), parse) == 7
    assert responder.sent == 1
    assert engine.usage()["cache_hits"] == 1


def test_key_depends_on_contract_and_schema(tmp_path: Path) -> None:
    """Изменив форму ответа, мы обязаны получить промах, а не старую запись."""
    engine = runner(tmp_path)
    first = engine.key(request(), attempt=1)
    other = engine.key(replace(request(), contract="covenant-formula-v3"), attempt=1)
    assert first != other


def test_retry_has_its_own_key(tmp_path: Path) -> None:
    """Повтор с тем же ключом прочитал бы ту запись, ради исправления которой начат."""
    engine = runner(tmp_path)
    assert engine.key(request(), attempt=1) != engine.key(request(), attempt=2)


def test_payload_change_changes_the_key(tmp_path: Path) -> None:
    engine = runner(tmp_path)
    assert engine.key(request(), attempt=1) != engine.key(request({"clause": "6.2"}), attempt=1)


# --- офлайн -------------------------------------------------------------------


def test_offline_miss_stops_before_the_network(tmp_path: Path) -> None:
    engine = runner(
        tmp_path,
        policy=CachePolicy.OFFLINE,
        offline=True,
        responder=(responder := Responder({"value": 1})),
    )

    with pytest.raises(CacheMissError):
        engine.run(request(), parse)
    assert responder.sent == 0


def test_offline_reads_what_is_cached(tmp_path: Path) -> None:
    warm = runner(tmp_path, responder=Responder({"value": 3}))
    warm.run(request(), parse)

    engine = runner(
        tmp_path,
        policy=CachePolicy.OFFLINE,
        offline=True,
        responder=(responder := Responder({"value": 99})),
    )
    assert engine.run(request(), parse) == 3
    assert responder.sent == 0


def test_offline_config_refuses_even_with_a_readable_cache(tmp_path: Path) -> None:
    """Гейт ключа срабатывает, даже если политика кэша промах пропустила."""
    engine = runner(
        tmp_path, policy=CachePolicy.READ_WRITE, offline=True, responder=Responder({"value": 1})
    )
    with pytest.raises(OfflineError):
        engine.run(request(), parse)


# --- бюджет -------------------------------------------------------------------


def test_live_call_limit_stops_the_run(tmp_path: Path) -> None:
    engine = runner(
        tmp_path, budget=Budget(max_live_calls=1), responder=(responder := Responder({"value": 1}))
    )

    engine.run(request({"clause": "6.1"}), parse)
    with pytest.raises(BudgetError):
        engine.run(request({"clause": "6.2"}), parse)
    assert responder.sent == 1


def test_transport_retries_count_against_the_limit(tmp_path: Path) -> None:
    """Повтор после таймаута стоит столько же, сколько первый запрос."""
    engine = runner(
        tmp_path,
        budget=Budget(max_live_calls=2),
        transport_retries=5,
        responder=(responder := Responder(TimeoutError("timeout"))),
    )

    with pytest.raises(BudgetError):
        engine.run(request(), parse)
    assert responder.sent == 2
    assert engine.budget.live_calls == 2
    assert from_runner(engine, Role.COMPILER).live_calls == 2


def test_input_token_limit_is_checked_before_sending(tmp_path: Path) -> None:
    engine = runner(
        tmp_path,
        budget=Budget(max_input_tokens_per_call=5),
        responder=(responder := Responder({"value": 1})),
    )

    with pytest.raises(BudgetError):
        engine.run(request({"clause": "x" * 500}), parse)
    assert responder.sent == 0


def test_total_input_token_limit_counts_multiple_calls() -> None:
    budget = Budget(max_total_input_tokens=10)
    budget.authorise(6)
    with pytest.raises(BudgetError):
        budget.authorise(5)


def test_request_estimate_includes_schema_and_retry_instruction() -> None:
    small = request()
    large = replace(
        small,
        schema={
            "type": "object",
            "properties": {f"field_{index}": {"type": "string"} for index in range(50)},
        },
    )

    assert large.estimated_input_tokens() > small.estimated_input_tokens()
    assert small.estimated_input_tokens(attempt=2) > small.estimated_input_tokens(attempt=1)


def test_output_limit_reaches_the_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Responses:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {"output_text": '{"value": 1}', "usage": None, "_request_id": "req-1"},
            )()

    class Client:
        responses = Responses()

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: Client())
    config = ModelConfig(name="test-model", api_key="sk-test", max_output_tokens=16000)

    send_to_openai(config, request(), "sk-test", 1)

    assert captured["max_output_tokens"] == 16000


def test_cost_limit_stops_the_next_call(tmp_path: Path) -> None:
    budget = Budget(
        max_output_tokens=20,
        max_estimated_cost=Decimal("0.002"),
        price_input_per_million=Decimal(10),
        price_output_per_million=Decimal(30),
    )
    engine = runner(tmp_path, budget=budget, responder=Responder({"value": 1}))

    engine.run(request({"clause": "6.1"}), parse)
    assert budget.estimated_cost > 0
    with pytest.raises(BudgetError):
        engine.run(request({"clause": "6.2"}), parse)


def test_model_price_overrides_the_global_fallback_price(tmp_path: Path) -> None:
    budget = Budget(
        price_input_per_million=Decimal(5),
        price_output_per_million=Decimal(30),
    )
    engine = runner(tmp_path, budget=budget, responder=Responder({"value": 1}))
    engine.config = replace(
        engine.config,
        price_input_per_million=Decimal("0.2"),
        price_output_per_million=Decimal("1.2"),
    )

    engine.run(request(), parse)

    assert budget.estimated_cost == Decimal("0.000044")


def test_cache_hits_do_not_spend_the_budget(tmp_path: Path) -> None:
    engine = runner(tmp_path, budget=Budget(max_live_calls=1), responder=Responder({"value": 5}))

    engine.run(request(), parse)
    assert engine.run(request(), parse) == 5


def test_budget_is_safe_under_concurrency(tmp_path: Path) -> None:
    """Два потока не должны проскочить проверку на последнем оставшемся вызове."""
    budget = Budget(max_live_calls=10)
    errors: list[BaseException] = []

    def spend() -> None:
        for _ in range(20):
            try:
                budget.authorise(1)
            except BudgetError:
                return
            except BaseException as exc:  # pragma: no cover — сюда попадать нечему
                errors.append(exc)
                return

    threads = [threading.Thread(target=spend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert budget.live_calls == 10


# --- повторы и разбор ---------------------------------------------------------


def test_invalid_answer_triggers_one_semantic_retry(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=(responder := Responder({"wrong": 1}, {"value": 42})))

    assert engine.run(request(), parse) == 42
    assert responder.sent == 2
    assert "KeyError" in responder.requests[1].instructions
    assert engine.usage()["invalid_responses"] == 1


def test_successful_semantic_retry_becomes_canonical_cache_entry(tmp_path: Path) -> None:
    first = runner(tmp_path, responder=Responder({"wrong": 1}, {"value": 42}))
    assert first.run(request(), parse) == 42

    second_responder = Responder(AssertionError("network must not be called"))
    second = runner(tmp_path, responder=second_responder)
    assert second.run(request(), parse) == 42
    assert second_responder.sent == 0
    assert [call.attempt for call in second.calls] == [1]
    assert second.calls[0].cache_hit


def test_two_invalid_answers_give_up(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=(responder := Responder({"wrong": 1})))

    with pytest.raises(InvalidResponseError):
        engine.run(request(), parse)
    assert responder.sent == 2


def test_semantic_failure_escalates_to_fallback_model(tmp_path: Path) -> None:
    primary_responder = Responder({"wrong": 1})
    fallback_responder = Responder({"value": 42})
    primary = runner(tmp_path / "primary", responder=primary_responder)
    primary.semantic_attempts = 1
    primary.config = replace(primary.config, name="gpt-5.6-terra")
    fallback = runner(tmp_path / "fallback", responder=fallback_responder)
    fallback.config = replace(fallback.config, name="gpt-5.6-sol")
    cascade = CascadingModelRunner(primary, fallback)

    assert cascade.run(request(), parse) == 42
    assert [call.model for call in cascade.calls] == ["gpt-5.6-terra", "gpt-5.6-sol"]
    assert (
        "основная модель не смогла выдать валидный ответ"
        in fallback_responder.requests[0].instructions
    )
    assert "KeyError" in fallback_responder.requests[0].instructions


def test_offline_primary_miss_replays_stable_fallback(tmp_path: Path) -> None:
    primary = runner(tmp_path / "primary", responder=Responder({"wrong": 1}))
    fallback = runner(tmp_path / "fallback", responder=Responder({"value": 42}))
    primary.semantic_attempts = 1
    primary.config = replace(primary.config, name="gpt-5.6-terra")
    fallback.config = replace(fallback.config, name="gpt-5.6-sol")
    cascade = CascadingModelRunner(primary, fallback)

    assert cascade.run(request(), parse) == 42

    primary.cache.policy = CachePolicy.OFFLINE
    fallback.cache.policy = CachePolicy.OFFLINE
    replay = CascadingModelRunner(primary, fallback)
    assert replay.run(request(), parse) == 42
    assert replay.fallback.calls[-1].cache_hit


def test_valid_but_incomplete_answer_escalates_to_fallback_model(tmp_path: Path) -> None:
    primary_responder = Responder({"value": 1})
    fallback_responder = Responder({"value": 42})
    cascade = CascadingModelRunner(
        runner(tmp_path / "primary", responder=primary_responder),
        runner(tmp_path / "fallback", responder=fallback_responder),
    )

    result = cascade.run(
        request(),
        parse,
        escalate_if=lambda value: "результат остался неполным" if value == 1 else None,
    )

    assert result == 42
    assert primary_responder.sent == 1
    assert fallback_responder.sent == 1
    assert "результат остался неполным" in fallback_responder.requests[0].instructions


def test_quota_error_does_not_escalate_to_fallback(tmp_path: Path) -> None:
    primary_responder = Responder(QuotaError("credit_balance_exhausted"))
    fallback_responder = Responder({"value": 42})
    cascade = CascadingModelRunner(
        runner(tmp_path / "primary", responder=primary_responder),
        runner(tmp_path / "fallback", responder=fallback_responder),
    )

    with pytest.raises(QuotaError):
        cascade.run(request(), parse)
    assert primary_responder.sent == 1
    assert fallback_responder.sent == 0


def test_semantic_attempt_limit_is_configurable(tmp_path: Path) -> None:
    responder = Responder({"wrong": 1}, {"still_wrong": 2}, {"value": 42})
    engine = runner(tmp_path, responder=responder)
    engine.semantic_attempts = 3

    assert engine.run(request(), parse) == 42
    assert [call.attempt for call in engine.calls] == [1, 2, 3]


def test_invalid_answer_is_not_cached(tmp_path: Path) -> None:
    """Иначе одна испорченная запись воспроизводила бы ошибку на каждом прогоне."""
    engine = runner(tmp_path, responder=Responder({"wrong": 1}))
    with pytest.raises(InvalidResponseError):
        engine.run(request(), parse)

    assert list(tmp_path.glob("*.json")) == []


def test_transport_error_is_retried_then_succeeds(tmp_path: Path) -> None:
    engine = runner(
        tmp_path, responder=(responder := Responder(TimeoutError("timeout"), {"value": 8}))
    )

    assert engine.run(request(), parse) == 8
    assert responder.sent == 2


def test_non_transport_error_is_not_retried(tmp_path: Path) -> None:
    """Ошибка запроса повтором не лечится, а тратит бюджет."""
    engine = runner(
        tmp_path, responder=(responder := Responder(ValueError("bad request: unknown parameter")))
    )

    with pytest.raises(ValueError, match="bad request"):
        engine.run(request(), parse)
    assert responder.sent == 1


def test_exhausted_credit_is_not_retried(tmp_path: Path) -> None:
    engine = runner(
        tmp_path,
        transport_retries=5,
        responder=(responder := Responder(QuotaError("credit_balance_exhausted"))),
    )

    with pytest.raises(QuotaError):
        engine.run(request(), parse)
    assert responder.sent == 1


def test_provider_rejection_is_visible_in_stage_telemetry(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=(responder := Responder(ProviderError("bad schema"))))

    with pytest.raises(ProviderError):
        engine.run(request(), parse)

    assert responder.sent == 1
    assert engine.budget.live_calls == 1
    assert engine.budget.estimated_cost == 0
    assert len(engine.calls) == 1
    record = engine.calls[0].record()
    assert record["provider_status"] == 400
    assert record["provider_error_code"] == "invalid_json_schema"
    assert record["request_id"] == "req-rejected"
    assert record["valid"] is False
    assert from_runner(engine, Role.COMPILER).live_calls == 1


# --- телеметрия и батчирование ------------------------------------------------


def test_telemetry_records_every_attempt(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=Responder({"wrong": 1}, {"value": 4}))
    engine.run(request(), parse)

    records = [call.record() for call in engine.calls]
    assert [item["attempt"] for item in records] == [1, 2]
    assert [item["valid"] for item in records] == [False, True]
    assert records[0]["role"] == "compiler"
    assert records[0]["account_id"] == "ACC-7801"
    assert records[1]["request_id"] == "req-2"
    assert records[0]["note"].startswith("KeyError")


def test_usage_separates_live_from_cached(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=Responder({"value": 1}))
    engine.run(request(), parse)
    engine.run(request(), parse)

    usage = engine.usage()
    assert usage == {
        "attempts": 2,
        "live_calls": 1,
        "cache_hits": 1,
        "invalid_responses": 0,
        "seconds_total": usage["seconds_total"],
        "budget": usage["budget"],
    }


def test_work_is_grouped_by_borrower() -> None:
    """Один запрос на заёмщика вместо запроса на пункт — условие выполнимости."""
    grouped = batch_by_account(
        [("ACC-1", "6.1"), ("ACC-2", "6.1"), ("ACC-1", "6.2"), ("ACC-1", "6.3")]
    )
    assert grouped == {"ACC-1": ["6.1", "6.2", "6.3"], "ACC-2": ["6.1"]}
