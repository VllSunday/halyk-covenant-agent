from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-sol"


class ConfigError(RuntimeError):
    pass


class OfflineError(RuntimeError):
    """Попытка сетевого вызова в офлайн-режиме."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Настройки одного вызова модели.

    Параметров семплирования здесь нет намеренно: у рассуждающих моделей их либо не
    принимают, либо они ни на что не влияют, а воспроизводимость прогона держится на
    кэше по содержимому запроса, а не на `temperature=0`.
    """

    name: str
    api_key: str | None
    reasoning_effort: str = "medium"
    timeout_seconds: float = 90.0
    max_retries: int = 2
    offline: bool = False

    def authorise_live_call(self, purpose: str) -> str:
        """Разрешение на сетевой вызов вместе с ключом.

        Единственная дверь наружу: клиента ни одного провайдера не собрать без ключа,
        а ключ выдаётся только здесь. Поэтому запрет, поставленный в этом методе,
        закрывает и те пути к сети, которые появятся в коде позже, — в отличие от
        проверки на стороне вызывающего, которую новый вызов просто не унаследует.
        """
        if self.offline:
            raise OfflineError(
                f"Офлайн-режим: {purpose} потребовал живого вызова {self.name}, "
                f"но в кэше ответа нет. Уберите --offline, когда будете готовы платить "
                f"за прогон."
            )
        if not self.api_key:
            raise ConfigError(
                f"Не задан ключ для модели {self.name}. Скопируйте .env.example в .env "
                f"и заполните его."
            )
        return self.api_key


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True, slots=True)
class Settings:
    compiler: ModelConfig
    ocr: ModelConfig
    classifier: ModelConfig
    verifier: ModelConfig
    artifacts_dir: Path
    max_concurrency: int
    # Шапка ответа. Значений по умолчанию нет намеренно: команда и почта уходят
    # организаторам, и подставленная за нас заглушка означала бы сдачу не от нас.
    team: str = ""
    contact_email: str = ""
    # Потолок расходов прогона. Пустое значение — без потолка: в боевом окне лимит
    # ставится осознанно, а тихое ограничение хуже отсутствующего.
    max_live_calls: int | None = None
    max_input_tokens: int | None = None
    max_cost_usd: Decimal | None = None
    # Цена за миллион токенов. Нужна только затем, чтобы потолок стоимости не был
    # декоративным: без неё расход прогона всегда равен нулю и лимит не сработает.
    price_input_per_million: Decimal = Decimal(0)
    price_output_per_million: Decimal = Decimal(0)
    offline: bool = False

    @classmethod
    def from_env(cls, env_file: Path | None = None, *, offline: bool = False) -> Settings:
        load_dotenv(env_file, override=False)
        api_key = os.getenv("OPENAI_API_KEY")
        offline = offline or os.getenv("HALYK_OFFLINE", "") not in ("", "0", "false", "no")
        cost = os.getenv("HALYK_MAX_COST_USD", "").strip()
        return cls(
            # Компиляция ковенанта — самый дорогой по последствиям шаг: ошибка здесь
            # ломает все три ячейки заёмщика сразу, поэтому уровень рассуждения выше.
            compiler=ModelConfig(
                name=os.getenv("HALYK_COMPILER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_COMPILER_EFFORT", "high"),
                offline=offline,
            ),
            # Распознавание — задача восприятия, а не рассуждения; высокий уровень
            # только замедляет её, не добавляя точности.
            ocr=ModelConfig(
                name=os.getenv("HALYK_OCR_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_OCR_EFFORT", "low"),
                offline=offline,
            ),
            # Отнесение операции к статье — задача чтения, а не вывода: высокий
            # уровень рассуждения удлиняет двенадцать батчей, не меняя ответа.
            classifier=ModelConfig(
                name=os.getenv("HALYK_CLASSIFIER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_CLASSIFIER_EFFORT", "medium"),
                offline=offline,
            ),
            verifier=ModelConfig(
                name=os.getenv("HALYK_VERIFIER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_VERIFIER_EFFORT", "xhigh"),
                offline=offline,
            ),
            artifacts_dir=Path(os.getenv("HALYK_ARTIFACTS_DIR", "artifacts")),
            max_concurrency=int(os.getenv("HALYK_MAX_CONCURRENCY", "4")),
            team=os.getenv("HALYK_TEAM", "").strip(),
            contact_email=os.getenv("HALYK_CONTACT_EMAIL", "").strip(),
            max_live_calls=_optional_int("HALYK_MAX_LIVE_CALLS"),
            max_input_tokens=_optional_int("HALYK_MAX_INPUT_TOKENS"),
            max_cost_usd=Decimal(cost) if cost else None,
            price_input_per_million=Decimal(os.getenv("HALYK_PRICE_INPUT_PER_MTOK", "0")),
            price_output_per_million=Decimal(os.getenv("HALYK_PRICE_OUTPUT_PER_MTOK", "0")),
            offline=offline,
        )

    def cache_dir(self, role: str) -> Path:
        """Каталог общего кэша под одну роль. Один на рабочий каталог, а не на прогон."""
        return self.artifacts_dir / "cache" / role

    def model_versions(self) -> dict[str, str]:
        """Идёт в run_manifest, поэтому имена берутся из тех же объектов, что и вызовы."""
        return {
            "covenant_compiler": self.compiler.name,
            "ocr": self.ocr.name,
            "classifier": self.classifier.name,
            "verifier": self.verifier.name,
        }
