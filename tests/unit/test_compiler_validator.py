"""Валидатор ответа компилятора.

Живых вызовов здесь нет и быть не может: проверяется не модель, а то, что мы
отказываемся принимать. Каждый тест подсовывает валидатору ровно один дефект.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from halyk.compiler.contract import (
    Cardinality,
    CompiledClause,
    FactRequirement,
    Resolution,
    Scope,
)
from halyk.compiler.validator import (
    ALLOWED_OPS,
    check_coverage,
    check_ledger_measures,
    check_requirements,
    check_threshold_form,
    check_unresolved_terms,
    resolve_requirements,
    validate,
)
from halyk.models import formula as ast
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.models.fact import OneOffPolicyFact
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
from halyk.money import Currency, Money

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


def need(kind: str, unit: Unit = Unit.MONEY, **overrides: object) -> FactRequirement:
    base: dict[str, object] = {
        "requirement_id": f"req-{kind}",
        "scenario_id": "P1",
        "account_id": "ACC-7801",
        "fact_kind": kind,
        "description": kind,
        "unit": unit,
        "period_start": date(2025, 1, 1),
        "period_end": date(2025, 12, 31),
    }
    return FactRequirement(**(base | overrides))  # type: ignore[arg-type]


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


def test_quote_may_be_supported_by_one_of_several_sources() -> None:
    support = document("support.pdf").model_copy(
        update={"pages": (PageFacts(number=3, text="дополнительное определение", char_count=25),)}
    )
    item = clause(formula={"source_refs": (source(), source(name="support.pdf"))})
    assert "quote_is_not_found" not in run(
        item, documents={"agreement.pdf": document(), "support.pdf": support}
    )


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
    item = clause(required_facts=(need("one_off_item"),))
    assert "fact_is_not_used" in codes(check_requirements(item))


def test_declared_fact_passes() -> None:
    item = clause(
        formula={
            "measure": Difference(
                left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                right=FactValue(fact_kind="one_off_item"),
            )
        },
        required_facts=(need("one_off_item"),),
    )
    assert check_requirements(item) == []


@pytest.mark.parametrize(
    ("kind", "description"),
    [
        ("borrower_revenue_ifrs", "audited revenue of the Borrower"),
        ("borrower_operating_expenses_ifrs", "Операционные расходы заёмщика"),
        ("borrower_audited_covenant_revenue", "revenue for covenant purposes"),
    ],
)
def test_borrower_ledger_measure_cannot_be_external_fact(kind: str, description: str) -> None:
    item = clause(required_facts=(need(kind, description=description),))
    errors = check_ledger_measures(item)
    assert codes(errors) == {"ledger_measure_declared_as_fact"}


def test_group_capex_remains_a_document_fact() -> None:
    item = clause(
        required_facts=(
            need(
                "group_capex",
                description="capital expenditure of the consolidated Group",
                scope=Scope.GROUP,
            ),
        )
    )
    assert check_ledger_measures(item) == []


def test_document_only_borrower_fact_remains_allowed() -> None:
    item = clause(required_facts=(need("one_off_policy", description="materiality threshold"),))
    assert check_ledger_measures(item) == []


def test_variable_threshold_is_rejected_in_favour_of_normalised_measure() -> None:
    item = clause(
        formula={
            "threshold": ast.Product(
                factors=(
                    Constant(value=Decimal("0.03")),
                    LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                )
            )
        }
    )
    assert codes(check_threshold_form(item)) == {"threshold_is_not_constant"}


# --- разрешение требований ----------------------------------------------------


def policy_fact(account: str = "ACC-7801") -> OneOffPolicyFact:
    return OneOffPolicyFact(
        account_id=account,
        source=source(),
        minimum=Money.from_decimal(300000, Currency.USD),
    )


def test_found_fact_becomes_resolved() -> None:
    item = clause(required_facts=(need("one_off_policy"),))
    resolved = resolve_requirements(item, [policy_fact()])
    assert resolved[0].resolution is Resolution.RESOLVED
    assert not resolved[0].is_open


def test_missing_fact_stays_open_for_the_next_step() -> None:
    """Ненайденная величина не обнуляется: она адрес следующего шага поиска."""
    item = clause(required_facts=(need("group_capex"),))
    resolved = resolve_requirements(item, [policy_fact()])
    assert resolved[0].resolution is Resolution.NEEDS_RESOLUTION
    assert item.model_copy(update={"required_facts": resolved}).open_requirements


def test_unresolved_terms_are_carried() -> None:
    """То, что модель не смогла перевести, обязано доехать до отчёта."""
    item = clause(unresolved_terms=("средневзвешенная стоимость капитала",))
    assert item.unresolved_terms == ("средневзвешенная стоимость капитала",)


def test_unresolved_terms_trigger_semantic_retry() -> None:
    item = clause(unresolved_terms=("Определение EBITDA Заёмщика",))
    assert codes(check_unresolved_terms(item)) == {"clause_has_unresolved_terms"}


# --- размерности --------------------------------------------------------------


def test_money_plus_ratio_is_rejected() -> None:
    """Формально валидное и бессмысленное дерево: исполнитель досчитал бы его."""
    assert "dimension_mismatch" in run(
        clause(
            formula={
                "measure": ast.Sum(
                    terms=(
                        LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                        Ratio(
                            numerator=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                            denominator=LedgerSum(selector=Selector(categories=(Cat.OPEX,))),
                        ),
                    )
                )
            }
        )
    )


def test_count_and_money_do_not_mix() -> None:
    assert "dimension_mismatch" in run(
        clause(
            formula={
                "measure": ast.Sum(
                    terms=(
                        LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                        ast.LedgerCount(selector=Selector(categories=(Cat.CAPEX,))),
                    )
                )
            }
        )
    )


def test_money_over_money_is_a_ratio() -> None:
    assert (
        run(
            clause(
                formula={
                    "measure": Ratio(
                        numerator=LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                        denominator=LedgerSum(selector=Selector(categories=(Cat.OPEX,))),
                    ),
                    "unit": Unit.RATIO,
                    "threshold": Constant(value=Decimal("0.5")),
                }
            )
        )
        == set()
    )


def test_declared_unit_must_match_the_tree() -> None:
    """Отношение, объявленное деньгами, уехало бы в ответ не в той единице."""
    assert "dimension_mismatch" in run(
        clause(
            formula={
                "measure": Ratio(
                    numerator=LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                    denominator=LedgerSum(selector=Selector(categories=(Cat.OPEX,))),
                ),
                "unit": Unit.MONEY,
            }
        )
    )


def test_fact_dimension_comes_from_the_requirement() -> None:
    """EBITDA плюс разовая статья: обе денежные, сложение законно."""
    assert (
        run(
            clause(
                formula={
                    "measure": ast.Sum(
                        terms=(
                            LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                            FactValue(fact_kind="one_off_item"),
                        )
                    )
                },
                required_facts=(need("one_off_item", Unit.MONEY),),
            )
        )
        == set()
    )


def test_fact_with_the_wrong_unit_breaks_the_sum() -> None:
    assert "dimension_mismatch" in run(
        clause(
            formula={
                "measure": ast.Sum(
                    terms=(
                        LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                        FactValue(fact_kind="coverage_share"),
                    )
                )
            },
            required_facts=(need("coverage_share", Unit.RATIO),),
        )
    )


def test_product_of_two_dimensional_values_is_rejected() -> None:
    assert "dimension_mismatch" in run(
        clause(
            formula={
                "measure": ast.Product(
                    factors=(
                        LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                        LedgerSum(selector=Selector(categories=(Cat.OPEX,))),
                    )
                )
            }
        )
    )


def test_scaling_money_by_a_constant_is_allowed() -> None:
    assert (
        run(
            clause(
                formula={
                    "measure": ast.Product(
                        factors=(
                            LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
                            Constant(value=Decimal("1.5")),
                        )
                    )
                }
            )
        )
        == set()
    )


# --- строгое сопоставление требований -----------------------------------------


def test_fact_of_another_borrower_does_not_resolve() -> None:
    """Совпадения по названию вида мало: счёт обязан совпасть тоже."""
    item = clause(required_facts=(need("one_off_policy"),))
    resolved = resolve_requirements(item, [policy_fact(account="ACC-9999")])
    assert resolved[0].resolution is Resolution.NEEDS_RESOLUTION


def test_two_matching_facts_are_ambiguous() -> None:
    """Выбор первого попавшегося зависел бы от порядка чтения файлов."""
    item = clause(required_facts=(need("one_off_policy", cardinality=Cardinality.ONE),))
    resolved = resolve_requirements(item, [policy_fact(), policy_fact()])
    assert resolved[0].resolution is Resolution.AMBIGUOUS
    assert resolved[0].is_open


def test_many_cardinality_accepts_several_facts() -> None:
    item = clause(required_facts=(need("one_off_policy", cardinality=Cardinality.MANY),))
    resolved = resolve_requirements(item, [policy_fact(), policy_fact()])
    assert resolved[0].resolution is Resolution.RESOLVED


def test_group_scope_is_carried_into_the_requirement() -> None:
    """P5/6.1: числитель по группе, знаменатель по заёмщику."""
    item = clause(required_facts=(need("ppe_roll_forward", scope=Scope.GROUP),))
    assert item.required_facts[0].scope is Scope.GROUP
