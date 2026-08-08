"""Удостоверение прогона. Что именно и чем считалось. Обоснование в docs/adr/0006."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RunMode(StrEnum):
    """Режим всегда виден в команде и попадает сюда, чтобы чужой прогон читался без догадок."""

    SOLVE = "solve"
    RESUME = "resume"
    REPLAY = "replay"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    mode: RunMode
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING

    input_sha256: str = Field(min_length=64, max_length=64)
    input_name: str

    git_commit: str | None = None
    git_dirty: bool = False
    python_version: str
    dependency_lock_sha256: str
    halyk_version: str

    models: dict[str, str]
    temperature: float = 0.0
    max_concurrency: int

    # Официальные исправления меняют ответ наравне с документами, поэтому версия
    # реестра фиксируется здесь: без неё прогон не воспроизводится.
    errata_sha256: str | None = None

    # Состав ячеек задаёт шаблон организаторов, а промпты и контракты определяют, что
    # ответит модель. И то, и другое меняет результат так же, как сами документы,
    # поэтому попадает в удостоверение прогона наравне с ними.
    template_sha256: str | None = None
    prompts_sha256: dict[str, str] = Field(default_factory=dict)

    submission_sha256: str | None = None
    model_cache_sha256: str | None = None

    replayed_from: str | None = None
