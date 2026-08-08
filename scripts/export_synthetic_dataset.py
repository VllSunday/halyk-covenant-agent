"""Выгружает синтетический англоязычный набор на диск.

Набор живёт в фикстурах сквозных тестов и собирается там на месте. На диск он нужен
ровно для одного: провести боевую проверку на данных, где заранее известен ответ, и не
трогать при этом ни открытый, ни приватный датасет организаторов.

Скрипт для разработки. В расчётный путь он не входит и ничего никуда не отправляет.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "tests")

from e2e import conftest as synthetic

DOCUMENTS = (
    ("e1-agreement.pdf", synthetic.E1_AGREEMENT),
    ("e1-superseded.pdf", synthetic.E1_SUPERSEDED),
    ("e1-kyc.pdf", synthetic.E1_KYC),
    ("e1-notes.pdf", synthetic.E1_NOTES),
    ("e2-agreement.pdf", synthetic.E2_AGREEMENT),
    ("e2-kyc.pdf", synthetic.E2_KYC),
    ("e2-notes.pdf", synthetic.E2_NOTES),
)


def export(target: Path, only: str | None = None) -> None:
    """Выгрузить набор. Страница-скан не выгружается: в тестах она проверяет общий
    кэш распознавания, а в боевой проверке стоила бы лишнего вызова ни за чем."""
    (target / "documents").mkdir(parents=True, exist_ok=True)
    for name, text in DOCUMENTS:
        synthetic.write_pdf(target / "documents" / name, text)
    (target / "master_ledger_2025.csv").write_text(synthetic.LEDGER, encoding="utf-8")

    template = json.loads(json.dumps(synthetic.TEMPLATE))
    if only is not None:
        if only not in template["answers"]:
            raise SystemExit(f"В наборе нет сценария {only}")
        template["answers"] = {only: template["answers"][only]}
    (target / "submission_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    args = sys.argv[1:]
    target = Path(args[0]) if args else Path("artifacts/synthetic")
    only = args[1] if len(args) > 1 else None
    export(target, only)
    scenarios = ", ".join(
        json.loads((target / "submission_template.json").read_text("utf-8"))["answers"]
    )
    print(f"Набор выгружен в {target}, сценарии: {scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
