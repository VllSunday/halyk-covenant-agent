"""Отнесение операций к статьям моделью. Батч — один заёмщик.

Модель видит описание, контрагента, знак суммы и дату, но не участвует в арифметике:
сумм в ответе нет вовсе. Она отвечает на единственный вопрос — какой статьёй договора
названа эта операция.

Ответ принимается только строго: чужой или пропущенный идентификатор операции делает
элемент недействительным. Повторяется при этом не весь заёмщик, а именно недостающие
строки — из-за одной испорченной записи терять полсотни разобранных незачем.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from halyk.config import ModelConfig
from halyk.hashing import sha256_payload
from halyk.llm.cache import ModelCache
from halyk.models.classification import TransactionCategory

OUTPUT_CONTRACT = "transaction-category-4"

CATEGORIES = tuple(c.value for c in TransactionCategory if c is not TransactionCategory.UNKNOWN)

INSTRUCTIONS = """\
Ты относишь операции заёмщика к статьям, которыми оперируют ковенанты кредитного
договора. Для каждой операции верни одну статью.

Статьи не вложены друг в друга и не пересекаются:
- revenue — выручка от основной деятельности, поступления от покупателей;
- opex — операционные расходы как строка отчётности: эксплуатация, обслуживание,
  ремонт и содержание производственного объекта («operating costs», «servicing»,
  «maintenance»). Сюда НЕ входят оплата труда, аренда, коммунальные услуги, налоги,
  проценты, страхование, а также маркетинг, реклама, аренда рекламных площадей,
  консультационные, юридические и аудиторские услуги;
- capex — капитальные затраты: приобретение и создание основных средств, передача
  оборудования. Капитализированные проценты сюда НЕ входят, они остаются
  процентными расходами;
- payroll — оплата труда и связанные с персоналом выплаты;
- utilities — электроэнергия, вода, газ, тепло, связь;
- rent — аренда и лизинговые платежи за помещения, площадки и оборудование;
- taxes — налоги, сборы, пошлины;
- interest_expense — проценты по заёмным средствам;
- insurance_premium — страховые премии;
- financing_inflow — поступления по привлечённому финансированию, а не выручка;
- other — операция не относится ни к одной из статей выше. Сюда идут маркетинг и
  реклама в любом виде, консультационные, юридические и аудиторские услуги, прочие
  административные и сбытовые расходы, а также полученные возвраты и скидки.

Правила:
- знак суммы важен: полученный возврат, скидка или зачёт по статье — не расход по
  ней; проценты полученные — не процентный расход, привлечённое финансирование — не
  выручка;
- относи по существу операции, а не по совпадению слова в описании: «outdoor
  marketing site hire» — это маркетинг, а не аренда;
- статья определяется её местом в отчётности заёмщика, а не бытовым смыслом:
  расход может быть операционным по смыслу и при этом не входить в строку
  «операционные расходы» договора;
- confidence — твоя уверенность от 0 до 1;
- evidence — дословный фрагмент описания, на котором основано решение;
- верни ровно по одному элементу на каждую переданную операцию, с тем же txn_id.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["txn_id", "category", "confidence", "evidence"],
                "properties": {
                    "txn_id": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class TransactionInput:
    """Что модель видит об операции."""

    txn_id: str
    description: str
    counterparty: str
    amount: str
    currency: str
    date: str

    def payload(self) -> dict[str, str]:
        return {
            "txn_id": self.txn_id,
            "description": self.description,
            "counterparty": self.counterparty,
            "amount": self.amount,
            "currency": self.currency,
            "date": self.date,
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class CategoryItem(BaseModel):
    """Элемент ответа модели. Проверяется полностью, а не по одному полю типа.

    Обоснование обязано быть куском самого описания. Это дешёвая защита от
    правдоподобной выдумки: модель, придумавшая основание, придумала и статью, а по
    одному значению `confidence` этого не видно.
    """

    model_config = ConfigDict(frozen=True)

    txn_id: str = Field(min_length=1)
    category: TransactionCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1)

    def grounded_in(self, description: str) -> bool:
        return _normalise(self.evidence) in _normalise(description)


@dataclass(frozen=True, slots=True)
class CategoryVerdict:
    txn_id: str
    category: TransactionCategory
    confidence: float
    evidence: str


@dataclass(slots=True)
class BatchCall:
    account_id: str
    size: int
    cache_hit: bool
    latency_seconds: float
    retry: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None
    missing: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "size": self.size,
            "cache_hit": self.cache_hit,
            "retry": self.retry,
            "latency_seconds": round(self.latency_seconds, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "request_id": self.request_id,
            "missing": list(self.missing),
        }


@dataclass(slots=True)
class CategoryClassifier:
    """Модель плюс кэш по содержимому батча."""

    config: ModelConfig
    cache: ModelCache
    calls: list[BatchCall] = field(default_factory=list)

    @property
    def live_calls(self) -> int:
        return sum(1 for call in self.calls if not call.cache_hit)

    def usage(self) -> dict[str, Any]:
        live = [c for c in self.calls if not c.cache_hit]
        return {
            "batches": len(self.calls),
            "live_calls": len(live),
            "cache_hits": len(self.calls) - len(live),
            "input_tokens": sum(c.input_tokens or 0 for c in live),
            "output_tokens": sum(c.output_tokens or 0 for c in live),
            "total_tokens": sum(c.total_tokens or 0 for c in live),
            "seconds_total": round(sum(c.latency_seconds for c in self.calls), 3),
            "seconds_slowest": round(max((c.latency_seconds for c in self.calls), default=0.0), 3),
        }

    def _key(self, rows: Sequence[TransactionInput], *, retry: bool = False) -> str:
        # Порядок строк на ответ не влияет, поэтому в ключ они входят отсортированными:
        # иначе перестановка дала бы промах на том же содержимом.
        #
        # Повтор всегда отдельная запись, даже когда состав строк совпал с первым
        # запросом: иначе он прочитал бы из кэша тот самый неполный ответ, ради
        # которого и затевался, и при повторном прогоне пропуск воспроизвёлся бы.
        return self.cache.key(
            model=self.config.name,
            params={
                "contract": OUTPUT_CONTRACT,
                "reasoning_effort": self.config.reasoning_effort,
                "categories": list(CATEGORIES),
                "attempt": "retry" if retry else "first",
            },
            system_prompt=INSTRUCTIONS,
            payload=[row.payload() for row in sorted(rows, key=lambda r: r.txn_id)],
        )

    def classify(
        self, account_id: str, rows: Sequence[TransactionInput]
    ) -> dict[str, CategoryVerdict]:
        """Разобрать батч, повторив ровно то, что не пришло с первого раза.

        Повтор один и только по недостающим строкам: он идёт отдельным запросом,
        поэтому и в кэш ложится отдельно — иначе при повторном прогоне пропуск
        воспроизводился бы вместе с ответом.
        """
        if not rows:
            return {}

        verdicts = self._attempt(account_id, rows)
        missing = [row for row in rows if row.txn_id not in verdicts]
        if missing:
            verdicts |= self._attempt(account_id, missing, retry=True)
        return verdicts

    def _attempt(
        self, account_id: str, rows: Sequence[TransactionInput], *, retry: bool = False
    ) -> dict[str, CategoryVerdict]:
        key = self._key(rows, retry=retry)
        started = time.perf_counter()
        if (cached := self.cache.get(key)) is not None:
            verdicts = self._verdicts(cached["items"], rows)
            self.calls.append(
                BatchCall(
                    account_id=account_id,
                    size=len(rows),
                    cache_hit=True,
                    retry=retry,
                    latency_seconds=time.perf_counter() - started,
                    missing=self._missing(verdicts, rows),
                )
            )
            return verdicts

        payload, usage, request_id = self._ask(rows)
        verdicts = self._verdicts(payload["items"], rows)
        self.calls.append(
            BatchCall(
                account_id=account_id,
                size=len(rows),
                cache_hit=False,
                retry=retry,
                latency_seconds=time.perf_counter() - started,
                input_tokens=usage[0],
                output_tokens=usage[1],
                total_tokens=usage[2],
                request_id=request_id,
                missing=self._missing(verdicts, rows),
            )
        )
        self.cache.put(key, payload)
        return verdicts

    def _ask(
        self, rows: Sequence[TransactionInput]
    ) -> tuple[dict[str, Any], tuple[int | None, int | None, int | None], str | None]:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(
            api_key=self.config.require_key(),
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        request: dict[str, Any] = {
            "model": self.config.name,
            "instructions": INSTRUCTIONS,
            "reasoning": {"effort": self.config.reasoning_effort},
            "input": json.dumps(
                [row.payload() for row in sorted(rows, key=lambda r: r.txn_id)],
                ensure_ascii=False,
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "transaction_categories",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        }
        response = client.responses.create(**request)
        parsed: dict[str, Any] = json.loads(response.output_text)
        usage = response.usage
        return (
            parsed,
            (
                usage.input_tokens if usage else None,
                usage.output_tokens if usage else None,
                usage.total_tokens if usage else None,
            ),
            getattr(response, "_request_id", None),
        )

    @staticmethod
    def _verdicts(
        items: Sequence[dict[str, Any]], rows: Sequence[TransactionInput]
    ) -> dict[str, CategoryVerdict]:
        """Принять только полностью пригодные элементы.

        Чужой идентификатор, повтор, значение вне диапазона и обоснование, которого
        нет в описании, — всё это одинаково делает элемент недействительным. Такая
        строка считается непришедшей и попадает в повтор.
        """
        by_id = {row.txn_id: row for row in rows}
        found: dict[str, CategoryVerdict] = {}
        for raw in items:
            try:
                item = CategoryItem.model_validate(raw)
            except ValidationError:
                continue
            source = by_id.get(item.txn_id)
            if source is None or item.txn_id in found or not item.grounded_in(source.description):
                continue
            found[item.txn_id] = CategoryVerdict(
                txn_id=item.txn_id,
                category=item.category,
                confidence=item.confidence,
                evidence=item.evidence,
            )
        return found

    @staticmethod
    def _missing(
        verdicts: dict[str, CategoryVerdict], rows: Sequence[TransactionInput]
    ) -> tuple[str, ...]:
        return tuple(sorted(row.txn_id for row in rows if row.txn_id not in verdicts))


def cache_signature() -> str:
    """Отпечаток промпта и контракта. Идёт в отчёт вместе с решениями."""
    return sha256_payload({"instructions": INSTRUCTIONS, "contract": OUTPUT_CONTRACT})[:16]
