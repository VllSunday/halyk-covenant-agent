"""Распознавание страниц, у которых нет пригодного текстового слоя.

OCR всегда идёт через кэш: повторный прогон обязан давать тот же текст, а платить за
распознавание одной и той же страницы дважды незачем. В ключ кэша входит не только
картинка, но и промпт с параметрами вызова — иначе правка промпта осталась бы без
эффекта, потому что ответ молча пришёл бы из кэша.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from halyk.config import ModelConfig
from halyk.hashing import sha256_bytes
from halyk.llm.cache import ModelCache

ImageDetail = Literal["low", "high", "auto", "original"]

# Версия контракта вывода. Меняется вместе с форматом, который мы ждём от модели.
OCR_OUTPUT_CONTRACT = "verbatim-markdown-1"

OCR_INSTRUCTIONS = """\
Ты распознаёшь страницу финансового документа. Верни её содержимое как текст.

Правила:
- переноси числа, суммы, валюты, проценты и даты ровно так, как они напечатаны,
  вместе с разделителями разрядов и знаками валюты;
- названия организаций переноси дословно, сохраняя знаки препинания и юридическую
  форму;
- таблицы передавай таблицами Markdown, сохраняя порядок строк и столбцов;
- сохраняй заголовки разделов и нумерацию пунктов;
- ничего не добавляй от себя, не исправляй и не пересчитывай;
- если фрагмент неразборчив, поставь на его месте [нечитаемо].

Верни только содержимое страницы, без пояснений о том, что ты сделал."""


class OcrError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OcrResponse:
    """Ответ движка. Кроме текста — всё, за что мы заплатили и по чему нас найдут.

    `request_id` нужен, когда страница распозналась неправдоподобно: без него у
    провайдера нечего спросить про конкретный вызов.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class OcrCall:
    """Одна распознанная страница: откуда, живым вызовом или из кэша, и почём."""

    document: str
    page: int
    cache_hit: bool
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "cache_hit": self.cache_hit,
            "latency_seconds": round(self.latency_seconds, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "request_id": self.request_id,
        }


def summarise_calls(calls: Sequence[OcrCall]) -> dict[str, Any]:
    """Свод по вызовам OCR за прогон.

    Токены считаются только по живым вызовам: попадание в кэш ничего не стоит, и
    складывать его с оплаченным расходом означало бы завышать бюджет на каждом
    повторном прогоне.
    """
    live = [call for call in calls if not call.cache_hit]
    return {
        "live_calls": len(live),
        "cache_hits": len(calls) - len(live),
        "input_tokens": sum(call.input_tokens or 0 for call in live),
        "output_tokens": sum(call.output_tokens or 0 for call in live),
        "total_tokens": sum(call.total_tokens or 0 for call in live),
        "seconds_total": round(sum(call.latency_seconds for call in calls), 3),
        "seconds_slowest": round(max((call.latency_seconds for call in calls), default=0.0), 3),
    }


class OcrEngine(Protocol):
    """Минимальный контракт движка: картинка на входе, текст и расход на выходе."""

    @property
    def name(self) -> str: ...

    @property
    def cache_signature(self) -> dict[str, Any]:
        """Всё, кроме картинки, что влияет на ответ. Попадает в ключ кэша."""
        ...

    def recognise(self, image: bytes) -> OcrResponse: ...


@dataclass(slots=True)
class CachedOcr:
    """Обёртка над движком, которая помнит уже распознанное.

    Ключ считается по байтам картинки, а не по имени файла и номеру страницы: так
    переименование или перестановка документов не приводит к повторным вызовам и,
    что важнее, не меняет результат. Имя и номер нужны только для отчёта — по ним
    видно, какая именно страница обошлась дороже или дольше всех.
    """

    engine: OcrEngine
    cache: ModelCache
    records: list[OcrCall] = field(default_factory=list)

    @property
    def live_calls(self) -> int:
        return sum(1 for call in self.records if not call.cache_hit)

    @property
    def cache_hits(self) -> int:
        return sum(1 for call in self.records if call.cache_hit)

    def usage(self) -> dict[str, Any]:
        return summarise_calls(self.records)

    def recognise(self, image: bytes, *, document: str, page: int) -> str:
        key = self.cache.key(
            model=self.engine.name,
            params=self.engine.cache_signature,
            system_prompt=OCR_INSTRUCTIONS,
            payload=sha256_bytes(image),
        )
        started = time.perf_counter()
        if (cached := self.cache.get(key)) is not None:
            self.records.append(
                OcrCall(
                    document=document,
                    page=page,
                    cache_hit=True,
                    latency_seconds=time.perf_counter() - started,
                )
            )
            return str(cached["text"])

        response = self.engine.recognise(image)
        self.records.append(
            OcrCall(
                document=document,
                page=page,
                cache_hit=False,
                latency_seconds=time.perf_counter() - started,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                request_id=response.request_id,
            )
        )
        self.cache.put(key, {"text": response.text})
        return response.text


@dataclass(slots=True)
class OpenAIVisionOcr:
    """Распознавание страницы моделью с поддержкой изображений.

    `detail="original"` передаёт картинку без понижения разрешения — на таблицах с
    мелкими числами это решает, прочитается ли `$251,338.94` или `$251,338.34`.
    """

    config: ModelConfig
    detail: ImageDetail = "original"

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def cache_signature(self) -> dict[str, Any]:
        return {
            "prompt_sha256": sha256_bytes(OCR_INSTRUCTIONS.encode()),
            "contract": OCR_OUTPUT_CONTRACT,
            "detail": self.detail,
            "reasoning_effort": self.config.reasoning_effort,
        }

    def recognise(self, image: bytes) -> OcrResponse:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(
            api_key=self.config.authorise_live_call("распознавание страницы"),
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        encoded = base64.b64encode(image).decode("ascii")
        # Запрос собирается словарём: перегрузки `responses.create` описаны через
        # TypedDict, и собрать их вручную здесь дороже, чем проверить ответ.
        request: dict[str, Any] = {
            "model": self.config.name,
            "instructions": OCR_INSTRUCTIONS,
            "reasoning": {"effort": self.config.reasoning_effort},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded}",
                            "detail": self.detail,
                        }
                    ],
                }
            ],
        }
        response = client.responses.create(**request)
        text = str(response.output_text or "")
        if not text.strip():
            raise OcrError(f"{self.config.name} вернул пустой текст страницы")

        usage = response.usage
        # `_request_id` — приватное поле, но это единственное место, где SDK отдаёт
        # идентификатор вызова у уже разобранного ответа; сохраняем его как есть.
        return OcrResponse(
            text=text,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            request_id=getattr(response, "_request_id", None),
        )


@dataclass(slots=True)
class MistralOcr:
    """Запасной движок на специализированном OCR-сервисе.

    Держим до тех пор, пока распознавание на основном провайдере не подтверждено
    прогоном: остаться в день соревнования с одним непроверенным путём хуже, чем
    нести лишний адаптер.
    """

    config: ModelConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def cache_signature(self) -> dict[str, Any]:
        return {"contract": OCR_OUTPUT_CONTRACT, "engine": "mistral"}

    def recognise(self, image: bytes) -> OcrResponse:
        from mistralai import Mistral  # noqa: PLC0415

        client = Mistral(api_key=self.config.authorise_live_call("распознавание страницы"))
        encoded = base64.b64encode(image).decode("ascii")
        response = client.ocr.process(
            model=self.config.name,
            document={"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
        )
        pages = getattr(response, "pages", None)
        if not pages:
            raise OcrError(f"{self.config.name} вернул ответ без страниц")
        # Специализированный OCR тарифицируется страницами, а не токенами, поэтому
        # счётчиков токенов у него нет — в отчёте останутся только время и признак кэша.
        return OcrResponse(text="\n".join(str(page.markdown) for page in pages))
