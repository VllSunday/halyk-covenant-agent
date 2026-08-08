"""Валидатор ответа компилятора.

Живых вызовов здесь нет и быть не может: проверяется не модель, а то, что мы
отказываемся принимать. Каждый тест подсовывает валидатору ровно один дефект.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from halyk.compiler.contract import CompiledClause, FactRequirement, Resolution
from halyk.compiler.validator import (
    ALLOWED_OPS,
    check_coverage,
    check_requirements,
    resolve_requirements,
    validate,
)
from halyk.models import formula as ast
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.models.formula import (
    Condition,
    Constant,
    CovenantFormula,
    Difference,
    Direction,
    ExternalMetric,
    FactValue,
    LedgerSum,
    Ratio,
    Selector,
)
from halyk.models.source import SourceRef

QUOTE = "капитальные затраты не превышают $300,000.00"
PAGE_TEXT = f"Статья 6 — Финансовые ковенанты\nПункт 6.1. За период {QUOTE} за год."
CELLS = [("P1", "6.1"), ("P1", "6.2")]


def document(
    name: str = "agreement.pdf", status: DocumentStatus = DocumentStatus.CURRENT
) -> DocumentFacts:
    return DocumentFacts(
        file_name=name,
        sha256="a" * 64,
        kind=DocumentKind.LOAN_AGREEMENT,
        status=status,
        account_id="ACC-7801",
        pages=(PageFacts(number=3, text=PAGE_TEXT, char_count=len(PAGE_TEXT)),),
    )


def source(name: str = "agreement.pdf", page: int = 3) -> SourceRef:
    return SourceRef(file_hash="a" * 64, file_name=name, page=page)


def clause(**overrides: object) -> CompiledClause:
    formula_fields: dict[str, object] = {
        "scenario_id": "P1",
        "clause_id": "6.1",
        "title": "капитальные затраты",
        "measure": LedgerSum(
            selector=Selector(categories=(Cat.CAPEX,), direction=Direction.OUTFLOW)
        ),
        "operator": Operator.LE,
        "threshold": Constant(value=Decimal("300000")),
        "unit": Unit.MONEY,
        "source_refs": (source(),),
        "quote": QUOTE,
        "confidence": 0.9,
    }
    formula_fields |= overrides.pop("formula", {})  # type: ignore[arg-type]
    base: dict[str, object] = {
        "formula": CovenantFormula(**formula_fields),  # type: ignore[arg-type]
        "period_start": date(2025, 1, 1),
        "period_end": date(2025, 12, 31),
    }
    return CompiledClause(**(base | overrides))  # type: ignore[arg-type]


def codes(errors: list) -> set[str]:  # type: ignore[type-arg]
    return {error.code for error in errors}


def run(item: CompiledClause, *, documents: dict[str, DocumentFacts] | None = None) -> set[str]:
    return codes(
        validate(
            item,
            documents={"agreement.pdf": document()} if documents is None else documents,
            expected=CELLS,
        )
    )


# --- исправный ответ ----------------------------------------------------------


def test_valid_clause_passes() -> None:
    assert run(clause()) == set()


def test_allowed_ops_cover_the_whole_ast() -> None:
    """Список операций закрыт, и он обязан совпадать с набором узлов дерева."""
    declared = {
        node.model_fields["op"].default
        for node in vars(ast).values()
        if isinstance(node, type) and "op" in getattr(node, "model_fields", {})
    }
    assert declared == ALLOWED_OPS


# --- ячейка -------------------------------------------------------------------


def test_unknown_cell_is_rejected() -> None:
    assert "cell_is_unknown" in run(clause(formula={"clause_id": "9.9"}))


def test_missing_and_duplicated_cells_are_reported() -> None:
    only_one = [clause(), clause()]
    found = codes(check_coverage(only_one, CELLS))
    assert found == {"cell_is_duplicated", "cell_is_missing"}


# --- цитата и источник --------------------------------------------------------


def test_paraphrased_quote_is_rejected() -> None:
    """Главная проверка: пересказ своими словами не проходит."""
    assert "quote_is_not_found" in run(clause(formula={"quote": "капзатраты не больше 300 тысяч"}))


def test_quote_survives_line_breaks() -> None:
    """В PDF фраза разорвана переносом, и это не повод отклонять ответ."""
    broken = PAGE_TEXT.replace("не превышают", "не\n   превышают")
    documents = {
        "agreement.pdf": document().model_copy(
            update={"pages": (PageFacts(number=3, text=broken, char_count=len(broken)),)}
        )
    }
    assert "quote_is_not_found" not in run(clause(), documents=documents)


def test_wrong_page_is_rejected() -> None:
    assert "page_is_missing" in run(clause(formula={"source_refs": (source(page=7),)}))


def test_unknown_file_is_rejected() -> None:
    assert "source_is_unknown" in run(clause(formula={"source_refs": (source(name="ghost.pdf"),)}))


def test_superseded_edition_is_rejected() -> None:
    """Пункт из вытесненной редакции даёт верное число при неверном пороге."""
    documents = {"agreement.pdf": document(status=DocumentStatus.SUPERSEDED)}
    assert "edition_is_superseded" in run(clause(), documents=documents)


# --- селектор и единицы -------------------------------------------------------


def test_open_selector_is_rejected() -> None:
    """Выборка без условий сложила бы весь реестр заёмщика в одну величину."""
    assert "selector_is_open" in run(clause(formula={"measure": LedgerSum(selector=Selector())}))


@pytest.mark.parametrize("month", [0, 13])
def test_impossible_month_is_rejected(month: int) -> None:
    assert "month_is_invalid" in run(
        clause(
            formula={
                "measure": LedgerSum(selector=Selector(categories=(Cat.REVENUE,), months=(month,)))
            }
        )
    )


def test_nested_selector_is_checked_too() -> None:
    """Проверка обходит дерево целиком, а не только его корень."""
    assert "selector_is_open" in run(
        clause(
            formula={
                "measure": Ratio(
                    numerator=LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                    denominator=LedgerSum(selector=Selector()),
                ),
                "unit": Unit.RATIO,
            }
        )
    )


def test_condition_is_checked_too() -> None:
    assert "selector_is_open" in run(
        clause(
            formula={
                "applies_when": Condition(
                    left=LedgerSum(selector=Selector()),
                    operator=Operator.GT,
                    right=Constant(value=Decimal(1)),
                )
            }
        )
    )


# --- требуемые факты ----------------------------------------------------------


def test_undeclared_fact_is_rejected() -> None:
    """Незаявленная величина остановила бы расчёт, но уже во время счёта."""
    assert "fact_is_not_declared" in run(
        clause(
            formula={
                "measure": Difference(
                    left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                    right=FactValue(fact_kind="one_off_item"),
                )
            }
        )
    )


def test_undeclared_external_metric_is_rejected() -> None:
    assert "fact_is_not_declared" in run(
        clause(
            formula={
                "measure": Ratio(
                    numerator=ExternalMetric(name="group_capex", description="капзатраты группы"),
                    denominator=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                ),
                "unit": Unit.RATIO,
            }
        )
    )


def test_declared_but_unused_fact_is_reported() -> None:
    item = clause(
        required_facts=(FactRequirement(fact_kind="one_off_item", description="разовые"),)
    )
    assert "fact_is_not_used" in codes(check_requirements(item))


def test_declared_fact_passes() -> None:
    item = clause(
        formula={
            "measure": Difference(
                left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                right=FactValue(fact_kind="one_off_item"),
            )
        },
        required_facts=(FactRequirement(fact_kind="one_off_item", description="разовые"),),
    )
    assert check_requirements(item) == []


# --- разрешение требований ----------------------------------------------------


def test_found_fact_becomes_resolved() -> None:
    item = clause(
        required_facts=(FactRequirement(fact_kind="one_off_policy", description="порог"),)
    )
    resolved = resolve_requirements(item, ["one_off_policy", "ppe_roll_forward"])
    assert resolved[0].resolution is Resolution.RESOLVED
    assert not resolved[0].is_open


def test_missing_fact_stays_open_for_the_next_step() -> None:
    """Ненайденная величина не обнуляется: она адрес следующего шага поиска."""
    item = clause(
        required_facts=(FactRequirement(fact_kind="group_capex", description="капзатраты группы"),)
    )
    resolved = resolve_requirements(item, ["one_off_policy"])
    assert resolved[0].resolution is Resolution.NEEDS_RESOLUTION
    assert item.model_copy(update={"required_facts": resolved}).open_requirements


def test_unresolved_terms_are_carried() -> None:
    """То, что модель не смогла перевести, обязано доехать до отчёта."""
    item = clause(unresolved_terms=("средневзвешенная стоимость капитала",))
    assert item.unresolved_terms == ("средневзвешенная стоимость капитала",)
