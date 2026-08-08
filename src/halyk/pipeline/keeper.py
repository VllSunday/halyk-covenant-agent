"""Последний ответ, который прошёл проверку целиком.

Строгий прогон не пишет неполный ответ, и это правильно. Но раньше он ещё и удалял
файл в начале работы — «файл принадлежит этому прогону». В боевом окне такое правило
означает, что неудачная попытка в 23:50 оставляет без ответа вообще, хотя удачная была
получаса назад.

Поэтому файл теперь не трогается до успеха, а рядом ведётся запись о том, какой прогон
его написал и с каким отпечатком. По ней видно, соответствует ли лежащий файл
последнему удачному прогону или его кто-то правил руками — а этого одним взглядом на
дату не понять.

Смешивать ячейки старого и нового ответа здесь нельзя и не делается: это отдельное
решение с отдельной ценой, и принимать его молча в момент падения — худший из
возможных моментов.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from halyk.config import Settings
from halyk.hashing import sha256_file

LAST_KNOWN_GOOD_NAME = "last_known_good.json"


@dataclass(frozen=True, slots=True)
class KnownGood:
    """Отметка о последнем полном ответе."""

    run_id: str
    path: str
    sha256: str
    recorded_at: str

    def record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "sha256": self.sha256,
            "recorded_at": self.recorded_at,
        }


def marker_path(settings: Settings) -> Path:
    return settings.artifacts_dir / LAST_KNOWN_GOOD_NAME


def remember(settings: Settings, *, run_id: str, output_path: Path, sha256: str) -> KnownGood:
    known = KnownGood(
        run_id=run_id,
        path=str(output_path),
        sha256=sha256,
        recorded_at=datetime.now(UTC).isoformat(),
    )
    target = marker_path(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(known.record(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return known


def recall(settings: Settings) -> KnownGood | None:
    path = marker_path(settings)
    if not path.exists():
        return None
    try:
        stored: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return KnownGood(
            run_id=str(stored["run_id"]),
            path=str(stored["path"]),
            sha256=str(stored["sha256"]),
            recorded_at=str(stored.get("recorded_at", "")),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def status(settings: Settings) -> dict[str, Any]:
    """Что известно о предыдущем полном ответе и цел ли он.

    Разбирается на четыре исхода, а не на «есть/нет»: пропавший и подменённый файлы
    выглядят одинаково для проверки на существование, но означают разное.
    """
    known = recall(settings)
    if known is None:
        return {"state": "absent", "detail": "полного ответа ещё не было"}

    path = Path(known.path)
    if not path.exists():
        return known.record() | {
            "state": "missing",
            "detail": f"файла {path} нет, хотя прогон {known.run_id} его записывал",
        }
    if sha256_file(path) != known.sha256:
        return known.record() | {
            "state": "modified",
            "detail": f"{path} отличается от того, что записал прогон {known.run_id}",
        }
    return known.record() | {
        "state": "intact",
        "detail": f"{path} — полный ответ прогона {known.run_id}",
    }


def describe(settings: Settings) -> str:
    """Одна строка для терминала и для текста отказа."""
    return str(status(settings)["detail"])
