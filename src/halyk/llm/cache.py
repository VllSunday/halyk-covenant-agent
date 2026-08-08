"""Кэш ответов моделей по содержимому запроса. Политика описана в docs/adr/0006 и 0010.

Кэш общий на рабочий каталог, а не на прогон: `halyk inventory --ocr` и `halyk solve`
задают модели один и тот же вопрос и обязаны получить один и тот же ответ, не
оплачивая его дважды. Безопасно это ровно потому, что ключ адресует содержимое: в него
входят модель, промпт, контракт, схема ответа, полезная нагрузка и хеши исходных
документов, — попадание означает, что этот самый запрос уже был задан.

Прогон при этом остаётся описуемым: журнал запоминает, каких записей он коснулся, и в
каталог прогона уходит индекс именно их, а не отпечаток всего каталога кэша.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from halyk.hashing import sha256_payload


class CacheRole(StrEnum):
    """Кто спрашивает — он же имя подкаталога кэша.

    Значения у классификатора исторические: под этими именами уже лежат оплаченные
    ответы, и переименование стоило бы их все.
    """

    OCR = "ocr"
    COMPILER = "compiler"
    RESOLVER = "resolver"
    CLASSIFIER = "categories"
    VERIFIER = "categories-verifier"


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


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Запись общего кэша, которой воспользовался прогон."""

    role: str
    key: str
    sha256: str
    reused: bool

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "key": self.key,
            "sha256": self.sha256,
            "source": "cache" if self.reused else "live",
        }


@dataclass(slots=True)
class CacheJournal:
    """Записи кэша, которых коснулся один прогон.

    Нужен затем, чтобы прогон оставался воспроизводимым при общем кэше. Отпечаток
    всего каталога менялся бы от чужой работы и не говорил бы ничего о нашей; здесь
    же в манифест уходит хеш ровно тех ответов, из которых собран этот ответ.
    """

    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def record(self, role: str, key: str, payload: Any, *, reused: bool) -> None:
        # Первое обращение побеждает: записанный этим прогоном ответ потом читается
        # из кэша, и перезапись пометила бы его как пришедший со стороны.
        self.entries.setdefault(
            key,
            CacheEntry(role=role, key=key, sha256=sha256_payload(payload), reused=reused),
        )

    @property
    def ordered(self) -> list[CacheEntry]:
        return sorted(self.entries.values(), key=lambda item: (item.role, item.key))

    @property
    def digest(self) -> str:
        return sha256_payload([[item.role, item.key, item.sha256] for item in self.ordered])

    def index(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "entries": [item.record() for item in self.ordered],
            "reused": sum(1 for item in self.ordered if item.reused),
            "written": sum(1 for item in self.ordered if not item.reused),
        }


@dataclass(slots=True)
class ModelCache:
    """Ключ собирается из всего, что влияет на ответ.

    Хеши исходных документов входят в ключ намеренно: иначе прогон на новом датасете
    попал бы в записи от старого, а именно этого мы и не хотим допустить.
    """

    directory: Path
    policy: CachePolicy = CachePolicy.WRITE_ONLY
    role: str = ""
    journal: CacheJournal | None = None
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
        if self.journal is not None:
            self.journal.record(self.role, key, loaded, reused=True)
        return loaded

    def put(self, key: str, response: dict[str, Any]) -> None:
        self.stats.live += 1
        self._path(key).write_text(
            json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        if self.journal is not None:
            self.journal.record(self.role, key, response, reused=False)

    def alias(self, source_key: str, target_key: str) -> None:
        """Сохранить успешный ответ под воспроизводимым альтернативным ключом."""
        source = self._path(source_key)
        if not source.exists():
            raise FileNotFoundError(source)
        target = self._path(target_key)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
