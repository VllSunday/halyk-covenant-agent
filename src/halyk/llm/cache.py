"""Кэш ответов моделей по содержимому запроса. Политика описана в docs/adr/0006."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from halyk.hashing import sha256_payload


class CachePolicy(StrEnum):
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"
    REPLAY = "replay"
    OFFLINE = "offline"


# REPLAY и OFFLINE оба останавливают прогон на промахе, но по разным причинам:
# первый повторяет конкретный прогон из его собственного каталога, второй работает с
# общим кэшем и просто не имеет права платить за живой вызов. Поэтому и текст ошибки
# у них разный — по нему видно, что чинить.


class CacheMissError(RuntimeError):
    """Промах там, где живой вызов запрещён. Молча уходить в сеть здесь нельзя."""


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    live: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.live


@dataclass(slots=True)
class ModelCache:
    """Ключ собирается из всего, что влияет на ответ.

    Хеши исходных документов входят в ключ намеренно: иначе прогон на новом датасете
    попал бы в записи от старого, а именно этого мы и не хотим допустить.
    """

    directory: Path
    policy: CachePolicy = CachePolicy.WRITE_ONLY
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        *,
        model: str,
        params: dict[str, Any],
        system_prompt: str,
        payload: Any,
        source_hashes: tuple[str, ...] = (),
    ) -> str:
        return sha256_payload(
            {
                "model": model,
                "params": params,
                "system": system_prompt,
                "payload": payload,
                "sources": sorted(source_hashes),
            }
        )

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.policy is CachePolicy.WRITE_ONLY:
            return None

        path = self._path(key)
        if not path.exists():
            if self.policy is CachePolicy.REPLAY:
                raise CacheMissError(
                    f"В кэше прогона нет ответа с ключом {key[:12]}. Повторить сабмит "
                    f"без живых вызовов невозможно — вероятно, каталог прогона неполный."
                )
            if self.policy is CachePolicy.OFFLINE:
                raise CacheMissError(
                    f"CACHE_MISS: ответа с ключом {key[:12]} в кэше нет, а --offline "
                    f"запрещает живой вызов. Либо запрос изменился с прошлого прогона, "
                    f"либо кэш неполный."
                )
            return None

        self.stats.hits += 1
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def put(self, key: str, response: dict[str, Any]) -> None:
        self.stats.live += 1
        self._path(key).write_text(
            json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
