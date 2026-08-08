"""Подготовка входа: архив или каталог, и шаблон ответа внутри него.

Организаторы обещали архив, но проверять решение удобнее на распакованном каталоге,
и оба входа обязаны давать один и тот же прогон. Различие живёт только здесь.

Шаблон ищется в самом датасете, а не задаётся флагом по умолчанию: состав ячеек —
часть входных данных, и подставленный со стороны шаблон означал бы, что мы отвечаем
на свой вопрос, а не на заданный.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

TEMPLATE_NAME = "submission_template.json"


class DatasetError(RuntimeError):
    """Вход не разобран: не тот формат, нет шаблона, архив с ловушкой в путях."""


def _is_safe(name: str) -> bool:
    """Путь внутри архива не выводит за каталог распаковки.

    Архив приходит извне, и запись по `../../` — известный способ подменить файл на
    машине, которая его распаковывает.
    """
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def unpack(input_path: Path, workspace: Path) -> Path:
    """Каталог датасета: сам вход, если это каталог, иначе распакованный архив."""
    if input_path.is_dir():
        return input_path
    if not zipfile.is_zipfile(input_path):
        raise DatasetError(
            f"{input_path} — не каталог и не zip-архив. Датасет ожидается в одном из этих видов"
        )

    target = workspace / "dataset"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path) as archive:
        unsafe = [item for item in archive.namelist() if not _is_safe(item)]
        if unsafe:
            raise DatasetError(f"{input_path}: пути ведут за пределы каталога: {unsafe[0]}")
        archive.extractall(target)
    return target


def find_template(root: Path, explicit: Path | None = None) -> Path:
    """Шаблон ответа: заданный флагом или единственный в датасете.

    Нескольких шаблонов быть не должно: выбирать между ними нечем, а ошибка выбора
    стоит всего состава ответа.
    """
    if explicit is not None:
        if not explicit.exists():
            raise DatasetError(f"Шаблона {explicit} нет")
        return explicit

    found = sorted(path for path in root.rglob(TEMPLATE_NAME) if path.is_file())
    if not found:
        raise DatasetError(f"В {root} нет {TEMPLATE_NAME}: состав ячеек ответа неизвестен")
    if len(found) > 1:
        names = ", ".join(path.as_posix() for path in found)
        raise DatasetError(f"В датасете больше одного шаблона: {names}")
    return found[0]
