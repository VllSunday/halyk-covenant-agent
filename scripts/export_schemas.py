"""Выгружает JSON Schema из моделей pydantic.

Схемы не пишутся руками: иначе они разъезжаются с кодом на второй неделе. Проверка
в CI перегенерирует их и сравнит с закоммиченными.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from halyk.models.adjustment import AdjustmentAdapter, NormalisedTransaction
from halyk.models.classification import ClassificationRecord
from halyk.models.covenant import CovenantIR
from halyk.models.fact import FactAdapter
from halyk.models.lineage import LineageRecord
from halyk.models.manifest import RunManifest
from halyk.models.metrics import RunMetrics
from halyk.models.submission import Submission

# Размеченные объединения выгружаются через TypeAdapter: схема одной ветки не
# описала бы артефакт, в котором лежат все.
EXPORTED: dict[str, type[BaseModel] | TypeAdapter[Any]] = {
    "adjustment": AdjustmentAdapter,
    "classification": ClassificationRecord,
    "covenant_ir": CovenantIR,
    "fact": FactAdapter,
    "lineage": LineageRecord,
    "normalised_transaction": NormalisedTransaction,
    "run_manifest": RunManifest,
    "metrics": RunMetrics,
    "submission": Submission,
}


def render(model: type[BaseModel] | TypeAdapter[Any]) -> str:
    schema = (
        model.json_schema(mode="serialization")
        if isinstance(model, TypeAdapter)
        else model.model_json_schema(mode="serialization")
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    check = "--check" in sys.argv
    target = Path("schemas")
    target.mkdir(exist_ok=True)

    stale = []
    for name, model in EXPORTED.items():
        path = target / f"{name}.schema.json"
        rendered = render(model)
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                stale.append(path)
        else:
            path.write_text(rendered, encoding="utf-8")

    if stale:
        print("Схемы разошлись с моделями:", *(f"  {p}" for p in stale), sep="\n")
        print("Выполните: make schemas")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
