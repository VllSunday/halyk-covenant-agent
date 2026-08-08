"""Единственная дверь к модели: кэш, бюджет, повторы и телеметрия.

Компилятор и resolver отличаются промптом и схемой ответа, а не способом вызова,
поэтому вызов один на всех. Универсального агентного фреймворка здесь нет и не
нужно — нужен предсказуемый вызов с проверяемым ответом.

Три вещи, ради которых модуль вообще существует.

Кэш спрашивается до создания клиента: в офлайне промах обязан остановить прогон, а
не привести к сетевому пути. Бюджет считает физические попытки, а не логические
батчи: транспортный повтор стоит столько же, сколько первый запрос, и лимит,
считающий батчи, защищает от перерасхода только на бумаге. Невалидный ответ не
ложится в кэш как обычный: иначе одна испорченная запись воспроизводила бы ошибку
на каждом последующем прогоне.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from halyk.config import ModelConfig
from halyk.hashing import sha256_payload
from halyk.llm.cache import ModelCache

T = TypeVar("T")

# Символов на токен. Оценка грубая и намеренно заниженная: она нужна до вызова,
# чтобы не начать запрос, который выйдет за лимит, а точное число приходит потом от
# провайдера и записывается в телеметрию.
_CHARS_PER_TOKEN = 3

# Ошибки, которые лечатся повтором того же запроса. Всё остальное — ошибка запроса
# или контракта, и повторять её бессмысленно.
_TRANSPORT_MARKERS = ("timeout", "429", "500", "502", "503", "504", "connection", "overloaded")


class Role(StrEnum):
    """Кто спрашивает. Идёт в ключ кэша и в телеметрию."""

    COMPILER = "compiler"
    RESOLVER = "resolver"
    CLASSIFIER = "classifier"
    VERIFIER = "verifier"


class BudgetError(RuntimeError):
    """Бюджет прогона исчерпан. Следующий запрос не отправляется."""


class InvalidResponseError(RuntimeError):
    """Модель ответила, но ответ не прошёл разбор ни с первого раза, ни с повтора."""


@dataclass(slots=True)
class Budget:
    """Потолок расходов прогона.

    Считает физические попытки, входные токены и деньги. Проверка идёт перед каждым
    обращением к сети, а не после: узнать о перерасходе постфактум — значит уже его
    допустить.
    """

    max_live_calls: int | None = None
    max_input_tokens: int | None = None
    max_estimated_cost: Decimal | None = None

    price_input_per_million: Decimal = Decimal(0)
    price_output_per_million: Decimal = Decimal(0)

    live_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: Decimal = Decimal(0)

    # Бюджет — общий ресурс, и проверка с записью обязаны быть неделимы. Иначе два
    # потока пройдут проверку на последнем оставшемся вызове и сделают два.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def authorise(self, estimated_input_tokens: int) -> None:
        """Разрешение на одну физическую попытку. Бросает, если её делать нельзя."""
        with self._lock:
            self._check(estimated_input_tokens)
            # Попытка засчитывается здесь, до сети: запрос, оборвавшийся по таймауту,
            # оплачен провайдером так же, как успешный.
            self.live_calls += 1
            self.input_tokens += estimated_input_tokens

    def _check(self, estimated_input_tokens: int) -> None:
        if self.max_live_calls is not None and self.live_calls >= self.max_live_calls:
            raise BudgetError(
                f"исчерпан лимит живых вызовов: {self.live_calls} из {self.max_live_calls}"
            )
        projected = self.input_tokens + estimated_input_tokens
        if self.max_input_tokens is not None and projected > self.max_input_tokens:
            raise BudgetError(
                f"запрос вышел бы за лимит входных токенов: {projected} из {self.max_input_tokens}"
            )
        if self.max_estimated_cost is not None and self.estimated_cost >= self.max_estimated_cost:
            raise BudgetError(
                f"исчерпан лимит стоимости: {self.estimated_cost} из {self.max_estimated_cost}"
            )

    def record(self, input_tokens: int | None, output_tokens: int | None, estimated: int) -> None:
        """Заменить оценку фактическими числами и досчитать стоимость."""
        with self._lock:
            if input_tokens is not None:
                self.input_tokens += input_tokens - estimated
            self.output_tokens += output_tokens or 0
            self.estimated_cost += (
                Decimal(input_tokens or estimated) * self.price_input_per_million
                + Decimal(output_tokens or 0) * self.price_output_per_million
            ) / Decimal(1_000_000)

    def report(self) -> dict[str, Any]:
        return {
            "live_calls": self.live_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": str(self.estimated_cost.quantize(Decimal("0.0001"))),
            "max_live_calls": self.max_live_calls,
            "max_input_tokens": self.max_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class Attempt:
    """Одна попытка: логическая при попадании в кэш, физическая при живом вызове."""

    role: Role
    account_id: str
    attempt: int
    cache_hit: bool
    latency_seconds: float
    valid: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    note: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "account_id": self.account_id,
            "attempt": self.attempt,
            "cache_hit": self.cache_hit,
            "latency_seconds": round(self.latency_seconds, 3),
            "valid": self.valid,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_id": self.request_id,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Request:
    """Всё, что определяет ответ, и потому целиком входит в ключ кэша."""

    role: Role
    account_id: str
    instructions: str
    schema: dict[str, Any]
    schema_name: str
    contract: str
    payload: Any
    source_hashes: tuple[str, ...] = ()

    def estimated_input_tokens(self) -> int:
        body = self.instructions + json.dumps(self.payload, ensure_ascii=False, sort_keys=True)
        return len(body) // _CHARS_PER_TOKEN


Answer = tuple[dict[str, Any], tuple[int | None, int | None], str | None]


class Sender(Protocol):
    """Как выглядит отправка запроса наружу."""

    def __call__(
        self, config: ModelConfig, request: Request, api_key: str, attempt: int
    ) -> Answer: ...


def send_to_openai(config: ModelConfig, request: Request, api_key: str, attempt: int) -> Answer:
    """Единственное место, где запрос уходит наружу.

    Вынесено из класса намеренно: это шов между нашим кодом и чужим сервисом. Тесты
    подставляют сюда свою функцию через поле `send`, а не подменяют метод — так
    граница видна в сигнатуре, а не в способе патчить.
    """
    from openai import OpenAI  # noqa: PLC0415

    # Повторы делает не SDK, а мы: иначе бюджет считал бы один вызов там, где
    # физически ушло три, и лимит защищал бы только на бумаге.
    client = OpenAI(api_key=api_key, timeout=config.timeout_seconds, max_retries=0)
    instructions = request.instructions
    if attempt > 1:
        instructions += (
            "\n\nПредыдущий ответ не прошёл разбор по схеме. Верни ответ строго по схеме, "
            "без пояснений вокруг него."
        )
    # Запрос собирается словарём: перегрузки `responses.create` описаны через
    # TypedDict, и собирать их вручную здесь дороже, чем проверить ответ.
    payload: dict[str, Any] = {
        "model": config.name,
        "instructions": instructions,
        "reasoning": {"effort": config.reasoning_effort},
        "input": json.dumps(request.payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": request.schema_name,
                "strict": True,
                "schema": request.schema,
            }
        },
    }
    response = client.responses.create(**payload)
    usage = response.usage
    return (
        json.loads(response.output_text),
        (usage.input_tokens if usage else None, usage.output_tokens if usage else None),
        getattr(response, "_request_id", None),
    )


def _is_transport_error(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in _TRANSPORT_MARKERS)


@dataclass(slots=True)
class StructuredModelRunner:
    """Вызов модели со структурированным ответом.

    Повторы двух разных природ и потому разведены. Транспортный — это тот же самый
    запрос после таймаута или 429, он не меняет ни промпта, ни ключа. Смысловой
    делается один раз и только тогда, когда ответ пришёл, но не разобрался; у него
    свой ключ кэша, иначе он прочитал бы ту самую запись, ради исправления которой
    и затевался.
    """

    config: ModelConfig
    cache: ModelCache
    budget: Budget
    transport_retries: int = 2
    # Отправка внедряется, а не берётся из класса: тест подставляет свою и доказывает,
    # что до сети дело не дошло, не полагаясь на возможность пропатчить метод.
    send: Sender = send_to_openai
    calls: list[Attempt] = field(default_factory=list)

    def key(self, request: Request, *, attempt: int) -> str:
        return self.cache.key(
            model=self.config.name,
            params={
                "role": request.role.value,
                "reasoning_effort": self.config.reasoning_effort,
                "contract": request.contract,
                "schema": sha256_payload(request.schema),
                "attempt": attempt,
            },
            system_prompt=request.instructions,
            payload=request.payload,
            source_hashes=request.source_hashes,
        )

    def run(self, request: Request, parse: Callable[[dict[str, Any]], T]) -> T:
        """Разобранный ответ модели или отказ.

        Порядок неслучаен: сначала кэш, потом бюджет, потом сеть. Промах в офлайне
        останавливает прогон в `cache.get`, до всякой попытки собрать клиента.
        """
        for attempt in (1, 2):
            key = self.key(request, attempt=attempt)
            if (cached := self.cache.get(key)) is not None:
                started = time.perf_counter()
                parsed, note = self._parse(cached, parse)
                self._log(request, attempt, True, started, parsed is not None, note=note)
                if parsed is not None:
                    return parsed
                # Испорченная запись — повод пойти на следующую попытку, а не сдаться:
                # у неё свой ключ и своё содержимое.
                continue

            payload, usage, request_id = self._ask(request, attempt)
            started = time.perf_counter()
            parsed, note = self._parse(payload, parse)
            self._log(
                request,
                attempt,
                False,
                started,
                parsed is not None,
                usage=usage,
                request_id=request_id,
                note=note,
            )
            if parsed is not None:
                # В кэш ложится только разобранное. Невалидный ответ, сохранённый
                # обычным образом, воспроизводил бы ошибку на каждом прогоне.
                self.cache.put(key, payload)
                return parsed

        raise InvalidResponseError(
            f"{request.role}/{request.account_id}: ответ не разобрался ни с первого раза, "
            f"ни с повтора"
        )

    def _parse(
        self, payload: dict[str, Any], parse: Callable[[dict[str, Any]], T]
    ) -> tuple[T | None, str]:
        try:
            return parse(payload), ""
        except Exception as exc:  # причина уходит в телеметрию целиком
            return None, f"{type(exc).__name__}: {exc}"[:300]

    def _ask(self, request: Request, attempt: int) -> Answer:
        """Живой вызов с транспортными повторами.

        Каждая попытка отдельно спрашивает бюджет: оборвавшийся по таймауту запрос
        провайдер тарифицирует так же, как успешный.
        """
        estimated = request.estimated_input_tokens()
        last: Exception | None = None

        for physical in range(self.transport_retries + 1):
            self.budget.authorise(estimated)
            api_key = self.config.authorise_live_call(f"{request.role} для {request.account_id}")
            try:
                payload, usage, request_id = self.send(self.config, request, api_key, attempt)
            except Exception as exc:  # решение принимает _is_transport_error
                last = exc
                if not _is_transport_error(exc) or physical == self.transport_retries:
                    raise
                continue
            self.budget.record(usage[0], usage[1], estimated)
            return payload, usage, request_id

        raise RuntimeError(f"транспортные повторы исчерпаны: {last}")

    def _log(
        self,
        request: Request,
        attempt: int,
        cache_hit: bool,
        started: float,
        valid: bool,
        *,
        usage: tuple[int | None, int | None] = (None, None),
        request_id: str | None = None,
        note: str = "",
    ) -> None:
        self.calls.append(
            Attempt(
                role=request.role,
                account_id=request.account_id,
                attempt=attempt,
                cache_hit=cache_hit,
                latency_seconds=time.perf_counter() - started,
                valid=valid,
                input_tokens=usage[0],
                output_tokens=usage[1],
                request_id=request_id,
                note=note,
            )
        )

    def usage(self) -> dict[str, Any]:
        live = [call for call in self.calls if not call.cache_hit]
        return {
            "attempts": len(self.calls),
            "live_calls": len(live),
            "cache_hits": len(self.calls) - len(live),
            "invalid_responses": sum(1 for call in self.calls if not call.valid),
            "seconds_total": round(sum(call.latency_seconds for call in self.calls), 3),
            "budget": self.budget.report(),
        }


def batch_by_account(items: Sequence[tuple[str, Any]]) -> dict[str, list[Any]]:
    """Сгруппировать работу по заёмщику.

    Один запрос на заёмщика вместо запроса на пункт — не оптимизация, а условие
    выполнимости: двенадцать батчей укладываются в боевое окно, четыре сотни
    одиночных запросов нет.
    """
    grouped: dict[str, list[Any]] = {}
    for account_id, item in items:
        grouped.setdefault(account_id, []).append(item)
    return grouped
