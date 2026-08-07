import ast
from pathlib import Path

import pytest

from halyk.ingest.guard import (
    GroundTruthLeakError,
    dataset_files,
    forbid_ground_truth,
    is_ground_truth,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "halyk"


def test_ground_truth_is_recognised(tmp_path: Path) -> None:
    assert is_ground_truth(tmp_path / "ground_truth.json")
    assert not is_ground_truth(tmp_path / "submission_template.json")


def test_forbid_ground_truth_raises(tmp_path: Path) -> None:
    with pytest.raises(GroundTruthLeakError):
        forbid_ground_truth(tmp_path / "ground_truth.json")


def test_forbid_passes_other_files_through(tmp_path: Path) -> None:
    ledger = tmp_path / "master_ledger_2025.csv"
    assert forbid_ground_truth(ledger) is ledger


def test_inventory_skips_the_key_and_is_ordered(tmp_path: Path) -> None:
    (tmp_path / "documents").mkdir()
    for name in ("ground_truth.json", "master_ledger_2025.csv", "documents/b.pdf"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "documents" / "a.pdf").write_text("x", encoding="utf-8")

    found = [p.relative_to(tmp_path).as_posix() for p in dataset_files(tmp_path)]
    assert found == ["documents/a.pdf", "documents/b.pdf", "master_ledger_2025.csv"]


def _toplevel_imports(path: Path) -> set[str]:
    """Только импорты уровня модуля: внутрифункциональные сюда осознанно не входят."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_production_modules_do_not_import_dev() -> None:
    """Пакет halyk.dev не должен попадать в дерево импортов расчётного пути.

    Проверяется статически, а не через sys.modules: иначе результат зависел бы от
    того, какой тест отработал раньше.
    """
    offenders = {
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path.parent.name != "dev"
        and any(name.startswith("halyk.dev") for name in _toplevel_imports(path))
    }
    assert offenders == set()
