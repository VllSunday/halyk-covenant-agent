from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    api_key: str | None
    temperature: float = 0.0

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
    verifier: ModelConfig
    artifacts_dir: Path
    max_concurrency: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file, override=False)
        return cls(
            compiler=ModelConfig(
                name=os.getenv("HALYK_COMPILER_MODEL", "gpt-5.6-sol"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            ocr=ModelConfig(
                name=os.getenv("HALYK_OCR_MODEL", "mistral-ocr-latest"),
                api_key=os.getenv("MISTRAL_API_KEY"),
            ),
            verifier=ModelConfig(
                name=os.getenv("HALYK_VERIFIER_MODEL", "claude-sonnet-5"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            ),
            artifacts_dir=Path(os.getenv("HALYK_ARTIFACTS_DIR", "artifacts")),
            max_concurrency=int(os.getenv("HALYK_MAX_CONCURRENCY", "8")),
        )

    def model_versions(self) -> dict[str, str]:
        """Идёт в run_manifest, поэтому имена берутся из тех же объектов, что и вызовы."""
        return {
            "covenant_compiler": self.compiler.name,
            "ocr": self.ocr.name,
            "verifier": self.verifier.name,
        }
