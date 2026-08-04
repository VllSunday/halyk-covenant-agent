from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Хеш каталога: по отсортированным относительным путям и содержимому файлов.

    Порядок обхода файловой системы не воспроизводится между машинами, поэтому пути
    сортируются явно, иначе один и тот же каталог даст разные хеши.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    """Каноническая форма для хеширования: отсортированные ключи, без лишних пробелов."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256_payload(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload))
