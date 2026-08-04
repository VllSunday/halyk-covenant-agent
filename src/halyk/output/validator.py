from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path("schemas")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    location: str
    message: str


def load_schema(name: str, schema_dir: Path = SCHEMA_DIR) -> dict[str, Any]:
    path = schema_dir / f"{name}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет схемы {path}")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def validate_document(document: Any, schema: dict[str, Any]) -> list[ValidationIssue]:
    """Возвращает все нарушения сразу, а не падает на первом.

    Перед сдачей важно увидеть полный список: править по одному в оставшееся время
    не получится.
    """
    validator = Draft202012Validator(schema)
    issues = []
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(part) for part in error.absolute_path) or "<корень>"
        issues.append(ValidationIssue(location=location, message=error.message))
    return issues


def validate_file(
    path: Path, schema_name: str, schema_dir: Path = SCHEMA_DIR
) -> list[ValidationIssue]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate_document(document, load_schema(schema_name, schema_dir))
