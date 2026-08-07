"""Часовой на входе датасета: ключ не должен попасть в расчётный путь.

Открытый набор поставляется вместе с `ground_truth.json`, и подсмотреть его можно
не только намеренно, но и случайно — достаточно перебрать все файлы каталога. Тогда
результат на публичных данных перестанет что-либо значить, а на приватных развалится.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

GROUND_TRUTH_FILENAMES = frozenset({"ground_truth.json"})


class GroundTruthLeakError(RuntimeError):
    pass


def is_ground_truth(path: Path) -> bool:
    return path.name.lower() in GROUND_TRUTH_FILENAMES


def forbid_ground_truth(path: Path) -> Path:
    """Пропускает путь дальше или падает.

    Ставится там, где файл читают адресно. Молчаливый пропуск здесь не годится:
    обращение к ключу из расчёта — дефект, который надо увидеть, а не сгладить.
    """
    if is_ground_truth(path):
        raise GroundTruthLeakError(
            f"{path} — ключ открытого датасета, он доступен только команде halyk score"
        )
    return path


def dataset_files(root: Path) -> tuple[Path, ...]:
    """Файлы датасета в детерминированном порядке, без ключа.

    Сортировка не косметика: порядок обхода файловой системы отличается между
    машинами, а прогон обязан воспроизводиться у организаторов до байта.
    """
    found: Iterable[Path] = (p for p in root.rglob("*") if p.is_file())
    return tuple(sorted((p for p in found if not is_ground_truth(p)), key=lambda p: p.as_posix()))
