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
    InvalidResponseError,
    Request,
    Role,
    StructuredModelRunner,
    batch_by_account,
)

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

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self.sent += 1
        outcome = self.outcomes[min(self.sent - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, (100, 20), f"req-{self.sent}"


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


def test_input_token_limit_is_checked_before_sending(tmp_path: Path) -> None:
    engine = runner(
        tmp_path,
        budget=Budget(max_input_tokens=5),
        responder=(responder := Responder({"value": 1})),
    )

    with pytest.raises(BudgetError):
        engine.run(request({"clause": "x" * 500}), parse)
    assert responder.sent == 0


def test_cost_limit_stops_the_next_call(tmp_path: Path) -> None:
    budget = Budget(
        max_estimated_cost=Decimal("0.0001"),
        price_input_per_million=Decimal(10),
        price_output_per_million=Decimal(30),
    )
    engine = runner(tmp_path, budget=budget, responder=Responder({"value": 1}))

    engine.run(request({"clause": "6.1"}), parse)
    assert budget.estimated_cost > 0
    with pytest.raises(BudgetError):
        engine.run(request({"clause": "6.2"}), parse)


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
    assert engine.usage()["invalid_responses"] == 1


def test_two_invalid_answers_give_up(tmp_path: Path) -> None:
    engine = runner(tmp_path, responder=(responder := Responder({"wrong": 1})))

    with pytest.raises(InvalidResponseError):
        engine.run(request(), parse)
    assert responder.sent == 2


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
