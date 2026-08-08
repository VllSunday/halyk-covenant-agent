from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from halyk.config import ModelConfig
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.llm.runner import Budget
from halyk.parsing.ocr import CachedOcr, OcrResponse, OpenAIVisionOcr


@dataclass
class FakeEngine:
    """Движок, который считает свои вызовы и отдаёт заранее известный текст."""

    text: str = "распознанный текст"
    signature: dict[str, Any] = field(default_factory=lambda: {"variant": "a"})
    calls: int = 0

    @property
    def name(self) -> str:
        return "fake-ocr"

    @property
    def cache_signature(self) -> dict[str, Any]:
        return self.signature

    def recognise(self, image: bytes) -> OcrResponse:
        self.calls += 1
        return OcrResponse(
            text=self.text,
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            request_id=f"req_{self.calls}",
        )


def cache(tmp_path: Path) -> ModelCache:
    return ModelCache(directory=tmp_path / "ocr", policy=CachePolicy.READ_WRITE)


def recognise(ocr: CachedOcr, image: bytes, page: int = 1) -> str:
    return ocr.recognise(image, document="b4.pdf", page=page)


def test_same_image_is_recognised_once(tmp_path: Path) -> None:
    engine = FakeEngine()
    ocr = CachedOcr(engine=engine, cache=cache(tmp_path))

    assert recognise(ocr, b"page-bytes") == "распознанный текст"
    assert recognise(ocr, b"page-bytes") == "распознанный текст"
    assert engine.calls == 1
    assert ocr.live_calls == 1
    assert ocr.cache_hits == 1


def test_different_images_are_recognised_separately(tmp_path: Path) -> None:
    engine = FakeEngine()
    ocr = CachedOcr(engine=engine, cache=cache(tmp_path))
    recognise(ocr, b"first", page=1)
    recognise(ocr, b"second", page=2)
    assert engine.calls == 2


def test_every_call_is_recorded_with_its_page_and_cost(tmp_path: Path) -> None:
    ocr = CachedOcr(engine=FakeEngine(), cache=cache(tmp_path))
    recognise(ocr, b"page-bytes", page=3)

    call = ocr.records[0]
    assert (call.document, call.page, call.cache_hit) == ("b4.pdf", 3, False)
    assert call.total_tokens == 1500
    assert call.request_id == "req_1"
    assert call.latency_seconds >= 0


def test_cache_hits_do_not_add_to_the_token_bill(tmp_path: Path) -> None:
    """Повторный прогон обязан показывать нулевой расход.

    Иначе бюджет прогона рос бы с каждым повтором, хотя провайдеру мы не платили.
    """
    shared = cache(tmp_path)
    first = CachedOcr(engine=FakeEngine(), cache=shared)
    recognise(first, b"page-bytes")
    assert first.usage()["total_tokens"] == 1500

    second = CachedOcr(engine=FakeEngine(), cache=shared)
    recognise(second, b"page-bytes")
    usage = second.usage()
    assert usage["live_calls"] == 0
    assert usage["cache_hits"] == 1
    assert usage["total_tokens"] == 0


def test_ocr_is_included_in_the_shared_cost_limit(tmp_path: Path) -> None:
    budget = Budget(
        price_input_per_million=Decimal(5),
        price_output_per_million=Decimal(30),
    )
    ocr = CachedOcr(
        engine=FakeEngine(),
        cache=cache(tmp_path),
        budget=budget,
        input_price=Decimal(2),
        output_price=Decimal(12),
    )

    recognise(ocr, b"page-bytes")

    assert budget.estimated_cost == Decimal("0.006")


def test_changed_prompt_parameters_invalidate_the_cache(tmp_path: Path) -> None:
    """Правка промпта или параметров вызова обязана пересчитать страницу.

    Иначе изменение осталось бы без эффекта: ответ молча пришёл бы из кэша, снятого
    старым промптом.
    """
    shared = cache(tmp_path)
    first = FakeEngine(signature={"detail": "original", "reasoning_effort": "low"})
    recognise(CachedOcr(engine=first, cache=shared), b"page")

    second = FakeEngine(signature={"detail": "original", "reasoning_effort": "high"})
    recognise(CachedOcr(engine=second, cache=shared), b"page")
    assert second.calls == 1


def test_openai_signature_covers_everything_that_changes_the_answer() -> None:
    engine = OpenAIVisionOcr(config=ModelConfig(name="gpt-5.6-sol", api_key=None))
    signature = engine.cache_signature
    assert set(signature) == {"prompt_sha256", "contract", "detail", "reasoning_effort"}
    assert signature["detail"] == "original"


def test_effort_reaches_the_signature() -> None:
    config = ModelConfig(name="gpt-5.6-sol", api_key=None, reasoning_effort="low")
    assert OpenAIVisionOcr(config=config).cache_signature["reasoning_effort"] == "low"
