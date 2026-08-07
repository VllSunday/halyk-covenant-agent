from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-sol"


class ConfigError(RuntimeError):
    pass


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

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                f"Не задан ключ для модели {self.name}. Скопируйте .env.example в .env "
                f"и заполните его."
            )
        return self.api_key


@dataclass(frozen=True, slots=True)
class Settings:
    compiler: ModelConfig
    ocr: ModelConfig
    classifier: ModelConfig
    verifier: ModelConfig
    artifacts_dir: Path
    max_concurrency: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file, override=False)
        api_key = os.getenv("OPENAI_API_KEY")
        return cls(
            # Компиляция ковенанта — самый дорогой по последствиям шаг: ошибка здесь
            # ломает все три ячейки заёмщика сразу, поэтому уровень рассуждения выше.
            compiler=ModelConfig(
                name=os.getenv("HALYK_COMPILER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_COMPILER_EFFORT", "high"),
            ),
            # Распознавание — задача восприятия, а не рассуждения; высокий уровень
            # только замедляет её, не добавляя точности.
            ocr=ModelConfig(
                name=os.getenv("HALYK_OCR_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_OCR_EFFORT", "low"),
            ),
            # Отнесение операции к статье — задача чтения, а не вывода: высокий
            # уровень рассуждения удлиняет двенадцать батчей, не меняя ответа.
            classifier=ModelConfig(
                name=os.getenv("HALYK_CLASSIFIER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_CLASSIFIER_EFFORT", "medium"),
            ),
            verifier=ModelConfig(
                name=os.getenv("HALYK_VERIFIER_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                reasoning_effort=os.getenv("HALYK_VERIFIER_EFFORT", "xhigh"),
            ),
            artifacts_dir=Path(os.getenv("HALYK_ARTIFACTS_DIR", "artifacts")),
            max_concurrency=int(os.getenv("HALYK_MAX_CONCURRENCY", "4")),
        )

    def model_versions(self) -> dict[str, str]:
        """Идёт в run_manifest, поэтому имена берутся из тех же объектов, что и вызовы."""
        return {
            "covenant_compiler": self.compiler.name,
            "ocr": self.ocr.name,
            "classifier": self.classifier.name,
            "verifier": self.verifier.name,
        }
