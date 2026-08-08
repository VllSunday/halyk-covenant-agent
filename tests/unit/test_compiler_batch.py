"""Компиляция пунктов заёмщика одним батчем.

Сети здесь нет. Отправка подменяется той же функцией, что и в тестах вызова модели, —
через поле `send`, а не патчем метода: так видно, что до сети дело не дошло, а не
что мы её обошли.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.compiler.batch import (
    CompilerBatch,
    CovenantCompiler,
    check_authority,
    check_period,
    normalise_derived_metrics,
    normalise_periods,
)
from halyk.compiler.contract import (
    CompiledClause,
    CompilerResponse,
    FactRequirement,
    Resolution,
)
from halyk.config import ModelConfig
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.llm.documents import BorrowerIsolationError
from halyk.llm.runner import Budget, InvalidResponseError, StructuredModelRunner
from halyk.llm.schema import strict_schema
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.models.fact import Fact, OneOffPolicyFact
from halyk.models.formula import (
    Constant,
    CovenantFormula,
    Difference,
    Direction,
    FactValue,
    LedgerSum,
    Selector,
    Sum,
)
from halyk.models.source import SourceRef
from halyk.money import Currency, Money

ACCOUNT = "ACC-7801"
QUOTE_61 = "капитальные затраты не превышают $300,000.00"
QUOTE_62 = "выручка за период составляет не менее $1,000,000.00"
PAGE_ONE = f"Статья 6 — Финансовые ковенанты\nПункт 6.1. За период {QUOTE_61} за год."
PAGE_TWO = f"Пункт 6.2. {QUOTE_62}"
PERIOD = (date(2025, 1, 1), date(2025, 12, 31))


def document(
    name: str = "agreement.pdf",
    *,
    account: str = ACCOUNT,
    kind: DocumentKind = DocumentKind.LOAN_AGREEMENT,
    status: DocumentStatus = DocumentStatus.CURRENT,
) -> DocumentFacts:
    return DocumentFacts(
        file_name=name,
        sha256=name.encode().hex().ljust(64, "0")[:64],
        kind=kind,
        status=status,
        account_id=account,
        pages=(
            PageFacts(number=1, text=PAGE_ONE, char_count=len(PAGE_ONE)),
            PageFacts(number=2, text=PAGE_TWO, char_count=len(PAGE_TWO)),
        ),
    )


def clause(
    clause_id: str = "6.1",
    *,
    quote: str = QUOTE_61,
    page: int = 1,
    file_name: str = "agreement.pdf",
    required_facts: tuple[FactRequirement, ...] = (),
    period: tuple[date, date] = PERIOD,
    measure: Any = None,
) -> CompiledClause:
    return CompiledClause(
        formula=CovenantFormula(
            scenario_id="P1",
            clause_id=clause_id,
            title=f"пункт {clause_id}",
            measure=measure
            or LedgerSum(selector=Selector(categories=(Cat.CAPEX,), direction=Direction.OUTFLOW)),
            operator=Operator.LE,
            threshold=Constant(value=Decimal("300000")),
            unit=Unit.MONEY,
            # Хеш здесь заведомо чужой: адрес источника всё равно берётся из переписи.
            source_refs=(SourceRef(file_hash="f" * 64, file_name=file_name, page=page),),
            quote=quote,
            confidence=0.9,
        ),
        period_start=period[0],
        period_end=period[1],
        required_facts=required_facts,
    )


def need(kind: str = "one_off_policy", **overrides: object) -> FactRequirement:
    base: dict[str, object] = {
        "requirement_id": f"req-{kind}",
        "scenario_id": "P1",
        "account_id": ACCOUNT,
        "fact_kind": kind,
        "description": kind,
        "unit": Unit.MONEY,
        "period_start": PERIOD[0],
        "period_end": PERIOD[1],
    }
    return FactRequirement(**(base | overrides))  # type: ignore[arg-type]


def with_fact(clause_id: str = "6.1") -> CompiledClause:
    """Пункт, которому не хватает величины из примечаний."""
    return clause(
        clause_id,
        measure=Difference(
            left=LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
            right=FactValue(fact_kind="one_off_policy"),
        ),
        required_facts=(need(),),
    )


def policy_fact() -> Fact:
    return OneOffPolicyFact(
        account_id=ACCOUNT,
        source=SourceRef(file_hash="a" * 64, file_name="notes.pdf", page=1),
        minimum=Money.from_decimal(50000, Currency.USD),
    )


def answer(*clauses: CompiledClause) -> dict[str, Any]:
    return {"clauses": [item.model_dump(mode="json") for item in clauses]}


class Responder:
    """Подменяет единственное место, где запрос уходит наружу."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.sent = 0

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self.sent += 1
        return self.outcomes[min(self.sent - 1, len(self.outcomes) - 1)], (100, 20), None


def compiler(tmp_path: Path, responder: Responder) -> CovenantCompiler:
    return CovenantCompiler(
        runner=StructuredModelRunner(
            config=ModelConfig(name="test-model", api_key="sk-test"),
            cache=ModelCache(directory=tmp_path, policy=CachePolicy.READ_WRITE),
            budget=Budget(),
            send=responder,
        )
    )


def batch(*documents: DocumentFacts, clauses: tuple[str, ...] = ("6.1", "6.2")) -> CompilerBatch:
    return CompilerBatch.build(ACCOUNT, "P1", clauses, documents or (document(),))


# --- валидный ответ -----------------------------------------------------------


def test_whole_borrower_is_compiled_in_one_call(tmp_path: Path) -> None:
    responder = Responder(answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)))
    result = compiler(tmp_path, responder).compile(batch())

    assert [item.cell for item in result.clauses] == [("P1", "6.1"), ("P1", "6.2")]
    assert responder.sent == 1
    assert result.period == PERIOD


def test_second_run_reads_the_cache(tmp_path: Path) -> None:
    responder = Responder(answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)))
    engine = compiler(tmp_path, responder)

    engine.compile(batch())
    engine.compile(batch())
    assert responder.sent == 1


# --- частичный ответ ----------------------------------------------------------


def test_missing_cell_costs_one_semantic_retry(tmp_path: Path) -> None:
    """Повтор делает Runner: у компилятора собственных вторых запросов нет."""
    responder = Responder(
        answer(clause("6.1")),
        answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)),
    )
    engine = compiler(tmp_path, responder)
    result = engine.compile(batch())

    assert len(result.clauses) == 2
    assert responder.sent == 2
    assert engine.runner.usage()["invalid_responses"] == 1
    assert [call.attempt for call in engine.runner.calls] == [1, 2]


def test_partial_answer_twice_gives_up(tmp_path: Path) -> None:
    responder = Responder(answer(clause("6.1")))
    with pytest.raises(InvalidResponseError):
        compiler(tmp_path, responder).compile(batch())
    assert responder.sent == 2


def test_rejection_reason_reaches_telemetry(tmp_path: Path) -> None:
    engine = compiler(tmp_path, Responder(answer(clause("6.1"))))
    with pytest.raises(InvalidResponseError):
        engine.compile(batch())

    assert "cell_is_missing" in engine.runner.calls[0].note


# --- неоднозначный ответ ------------------------------------------------------


def test_two_matching_facts_leave_the_requirement_open(tmp_path: Path) -> None:
    """Выбор первого попавшегося зависел бы от порядка чтения файлов."""
    responder = Responder(answer(with_fact("6.1"), with_fact("6.2")))
    result = compiler(tmp_path, responder).compile(batch(), facts=[policy_fact(), policy_fact()])

    requirement = result.clauses[0].required_facts[0]
    assert requirement.resolution is Resolution.AMBIGUOUS
    assert len(result.open_requirements) == 2


def test_single_matching_fact_closes_the_requirement(tmp_path: Path) -> None:
    responder = Responder(answer(with_fact("6.1"), with_fact("6.2")))
    result = compiler(tmp_path, responder).compile(batch(), facts=[policy_fact()])

    assert result.clauses[0].required_facts[0].resolution is Resolution.RESOLVED
    assert result.open_requirements == ()


def test_unfound_fact_becomes_work_for_the_resolver(tmp_path: Path) -> None:
    responder = Responder(answer(with_fact("6.1"), with_fact("6.2")))
    result = compiler(tmp_path, responder).compile(batch())

    assert [item.requirement_id for item in result.open_requirements] == [
        "req-one_off_policy",
        "req-one_off_policy",
    ]


# --- невалидный ответ ---------------------------------------------------------


def test_paraphrased_quote_is_not_compiled(tmp_path: Path) -> None:
    responder = Responder(
        answer(clause("6.1", quote="капзатраты не больше 300 тысяч"), clause("6.2", page=2))
    )
    with pytest.raises(InvalidResponseError):
        compiler(tmp_path, responder).compile(batch())
    assert responder.sent == 2


def test_invalid_answer_is_not_cached(tmp_path: Path) -> None:
    responder = Responder(answer(clause("6.1", quote="пересказ")))
    with pytest.raises(InvalidResponseError):
        compiler(tmp_path, responder).compile(batch())
    assert list(tmp_path.glob("*.json")) == []


def test_answer_that_is_not_a_response_is_rejected(tmp_path: Path) -> None:
    responder = Responder({"clauses": "шесть-один"})
    with pytest.raises(InvalidResponseError):
        compiler(tmp_path, responder).compile(batch())


def test_unknown_file_is_rejected(tmp_path: Path) -> None:
    responder = Responder(
        answer(clause("6.1", file_name="ghost.pdf"), clause("6.2", quote=QUOTE_62, page=2))
    )
    engine = compiler(tmp_path, responder)
    with pytest.raises(InvalidResponseError):
        engine.compile(batch())
    assert "source_is_unknown" in engine.runner.calls[0].note


# --- адрес источника принадлежит нам ------------------------------------------


def test_source_is_rebuilt_from_the_inventory(tmp_path: Path) -> None:
    """Хеш и статус документа мы знаем сами, от модели берём только имя и страницу."""
    responder = Responder(answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)))
    result = compiler(tmp_path, responder).compile(batch())

    ref = result.clauses[0].formula.source_refs[0]
    assert ref.file_hash == document().sha256
    assert ref.kind is DocumentKind.LOAN_AGREEMENT
    assert ref.account_id == ACCOUNT


def test_superseded_edition_cannot_be_claimed_current(tmp_path: Path) -> None:
    """Ответ объявляет редакцию действующей, перепись — вытесненной; верим переписи."""
    responder = Responder(answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)))
    engine = compiler(tmp_path, responder)
    old = document(status=DocumentStatus.SUPERSEDED)

    with pytest.raises(InvalidResponseError):
        engine.compile(batch(old))
    assert "edition_is_superseded" in engine.runner.calls[0].note


def test_draft_report_may_not_carry_a_clause(tmp_path: Path) -> None:
    """Промежуточная ведомость читается и хранится, но порога из неё мы не берём."""
    draft = document(kind=DocumentKind.AUDIT_PROCEDURES, status=DocumentStatus.DRAFT)
    responder = Responder(answer(clause("6.1"), clause("6.2", quote=QUOTE_62, page=2)))
    engine = compiler(tmp_path, responder)

    with pytest.raises(InvalidResponseError):
        engine.compile(batch(draft))
    assert "source_is_not_authoritative" in engine.runner.calls[0].note


def test_authority_check_reads_the_ref_and_not_the_document() -> None:
    """Право документа уже проставлено в адресе — проверке остаётся его прочитать."""
    assert [error.code for error in check_authority(clause())] == ["source_is_not_authoritative"]


# --- изоляция и детерминированность -------------------------------------------


def test_foreign_document_stops_the_batch() -> None:
    """Молчаливый фильтр превратил бы ошибку привязки в пропавшую величину."""
    with pytest.raises(BorrowerIsolationError, match=r"other\.pdf"):
        CompilerBatch.build(
            ACCOUNT, "P1", ["6.1"], [document(), document("other.pdf", account="ACC-9999")]
        )


def test_input_order_does_not_change_the_request(tmp_path: Path) -> None:
    """Порядок обхода каталога не воспроизводится, а он вошёл бы в ключ кэша."""
    first = document("a.pdf")
    second = document("b.pdf")
    engine = compiler(tmp_path, Responder(answer(clause("6.1"))))

    straight = CompilerBatch.build(ACCOUNT, "P1", ["6.1", "6.2"], [first, second])
    reversed_ = CompilerBatch.build(ACCOUNT, "P1", ["6.2", "6.1"], [second, first])
    assert straight.payload() == reversed_.payload()
    assert engine.runner.key(engine.request(straight), attempt=1) == engine.runner.key(
        engine.request(reversed_), attempt=1
    )


def test_documents_of_another_dataset_miss_the_cache(tmp_path: Path) -> None:
    engine = compiler(tmp_path, Responder(answer(clause("6.1"))))
    same_text = document().model_copy(update={"sha256": "b" * 64})

    assert engine.runner.key(engine.request(batch()), attempt=1) != engine.runner.key(
        engine.request(batch(same_text)), attempt=1
    )


# --- период -------------------------------------------------------------------


def test_clauses_of_one_borrower_share_the_period() -> None:
    other = clause("6.2", quote=QUOTE_62, page=2, period=(date(2024, 1, 1), date(2024, 12, 31)))
    assert [error.code for error in check_period([clause(), other])] == ["period_is_inconsistent"]


def test_single_period_outlier_is_normalised_by_majority() -> None:
    outlier = clause("6.3", period=(date(2024, 1, 1), date(2024, 12, 31)))
    clauses = normalise_periods((clause("6.1"), clause("6.2"), outlier))
    assert {(item.period_start, item.period_end) for item in clauses} == {PERIOD}


def test_periods_without_a_majority_stay_inconsistent() -> None:
    other = clause("6.2", period=(date(2024, 1, 1), date(2024, 12, 31)))
    clauses = normalise_periods((clause("6.1"), other))
    assert [error.code for error in check_period(clauses)] == ["period_is_inconsistent"]


def test_undisclosed_ebitda_is_expanded_from_the_ledger() -> None:
    item = clause(
        measure=FactValue(fact_kind="ebitda"),
        required_facts=(need("ebitda"),),
    )
    normalised = normalise_derived_metrics((item,), (document(),))[0]
    assert isinstance(normalised.formula.measure, Difference)
    assert normalised.required_facts == ()


def test_explicitly_disclosed_ebitda_remains_a_document_fact() -> None:
    item = clause(
        measure=FactValue(fact_kind="ebitda"),
        required_facts=(need("ebitda"),),
    )
    notes = document("notes.pdf", kind=DocumentKind.FINANCIAL_NOTES).model_copy(
        update={
            "pages": (
                PageFacts(
                    number=1,
                    text="EBITDA for the period amounts to $300,000.00.",
                    char_count=47,
                ),
            )
        }
    )
    normalised = normalise_derived_metrics((item,), (document(), notes))[0]
    assert isinstance(normalised.formula.measure, FactValue)
    assert normalised.required_facts == (item.required_facts[0],)


def test_inline_ebitda_does_not_double_count_detailed_expenses() -> None:
    detailed = Difference(
        left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,), direction=Direction.INFLOW)),
        right=Sum(
            terms=(
                LedgerSum(selector=Selector(categories=(Cat.OPEX,), direction=Direction.OUTFLOW)),
                LedgerSum(
                    selector=Selector(categories=(Cat.PAYROLL,), direction=Direction.OUTFLOW)
                ),
            )
        ),
    )
    item = clause(measure=detailed).model_copy(
        update={
            "formula": clause(measure=detailed).formula.model_copy(
                update={"title": "Нагрузка к EBITDA"}
            )
        }
    )

    normalised = normalise_derived_metrics((item,), (document(),))[0]

    assert isinstance(normalised.formula.measure, Difference)
    assert isinstance(normalised.formula.measure.right, LedgerSum)
    assert normalised.formula.measure.right.selector.categories == (Cat.OPEX,)


def test_non_ebitda_difference_is_not_rewritten() -> None:
    detailed = Difference(
        left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,), direction=Direction.INFLOW)),
        right=Sum(
            terms=(
                LedgerSum(selector=Selector(categories=(Cat.OPEX,), direction=Direction.OUTFLOW)),
                LedgerSum(
                    selector=Selector(categories=(Cat.PAYROLL,), direction=Direction.OUTFLOW)
                ),
            )
        ),
    )
    item = clause(measure=detailed)

    normalised = normalise_derived_metrics((item,), (document(),))[0]

    assert normalised.formula.measure == detailed


def test_inconsistent_period_is_not_compiled(tmp_path: Path) -> None:
    responder = Responder(
        answer(
            clause("6.1"),
            clause("6.2", quote=QUOTE_62, page=2, period=(date(2024, 1, 1), date(2024, 12, 31))),
        )
    )
    with pytest.raises(InvalidResponseError):
        compiler(tmp_path, responder).compile(batch())


# --- схема запроса ------------------------------------------------------------


def nodes(schema: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        found.append(schema)
        for key, value in schema.items():
            children = value.values() if key in ("properties", "$defs") else [value]
            for child in children:
                found += nodes(child)
    elif isinstance(schema, list):
        for item in schema:
            found += nodes(item)
    return found


def test_schema_has_no_optional_fields() -> None:
    """Строгий режим необязательных полей не знает: любое отсутствие — отказ схемы."""
    for node in nodes(strict_schema(CompilerResponse)):
        if "properties" in node:
            assert set(node["required"]) == set(node["properties"])
            assert node["additionalProperties"] is False


def test_schema_carries_no_defaults_and_no_oneof() -> None:
    text = repr(strict_schema(CompilerResponse))
    assert "'default'" not in text
    assert "'oneOf'" not in text
    assert "'discriminator'" not in text


def test_recursive_tree_survives_hardening() -> None:
    """Дерево формулы ссылается само на себя, и переписывание схемы не должно его рвать."""
    terms = strict_schema(CompilerResponse)["$defs"]["Sum"]["properties"]["terms"]
    assert {"$ref": "#/$defs/Sum"} in terms["items"]["anyOf"]
