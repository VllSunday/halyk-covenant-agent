"""Дочитывание открытых требований одним батчем.

Сети нет: отправка подменяется, кэш живёт во временном каталоге. Проверяется не
модель, а то, что мы отказываемся принимать, и то, чем отличается спорный ответ от
неполного.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from halyk.compiler.contract import Cardinality, FactRequirement, Resolution
from halyk.config import ModelConfig
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.llm.documents import BorrowerIsolationError
from halyk.llm.runner import Budget, InvalidResponseError, Role, StructuredModelRunner
from halyk.llm.schema import strict_schema
from halyk.models.adjustment import AdjustmentAction, AdjustmentStatus, ReclassifyAdjustment
from halyk.models.covenant import Unit
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.money import Currency
from halyk.resolution.batch import RequirementResolver, ResolverBatch
from halyk.resolution.contract import (
    Derivation,
    DerivationTerm,
    Evidence,
    ProposedAdjustment,
    ResolvedFact,
    ResolverResponse,
    Sign,
)
from halyk.resolution.validator import validate

ACCOUNT = "ACC-7801"
FLOOR = "разовой признаётся статья не менее $50,000.00"
MOVEMENT = (
    "Основные средства: opening $1,200,000.00, депрециация $150,000.00, "
    "выбытий не было, closing $1,450,000.00"
)
PAGE_ONE = f"Примечание 4. {FLOOR}"
PAGE_TWO = f"Примечание 7. {MOVEMENT}"
PERIOD = (date(2025, 1, 1), date(2025, 12, 31))


def document(
    name: str = "notes.pdf",
    *,
    account: str = ACCOUNT,
    kind: DocumentKind = DocumentKind.FINANCIAL_NOTES,
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


def need(
    requirement_id: str = "req-1",
    *,
    kind: str = "one_off_policy",
    unit: Unit = Unit.MONEY,
    **overrides: object,
) -> FactRequirement:
    base: dict[str, object] = {
        "requirement_id": requirement_id,
        "scenario_id": "P1",
        "account_id": ACCOUNT,
        "fact_kind": kind,
        "description": kind,
        "unit": unit,
        "currency": Currency.USD,
        "period_start": PERIOD[0],
        "period_end": PERIOD[1],
    }
    return FactRequirement(**(base | overrides))  # type: ignore[arg-type]


def found(
    requirement_id: str = "req-1",
    *,
    kind: str = "one_off_policy",
    amount: str = "50000",
    unit: Unit = Unit.MONEY,
    currency: Currency | None = Currency.USD,
    quote: str = FLOOR,
    page: int = 1,
    file_name: str = "notes.pdf",
) -> ResolvedFact:
    return ResolvedFact(
        requirement_id=requirement_id,
        fact_kind=kind,
        amount=Decimal(amount),
        unit=unit,
        currency=currency,
        evidence=(Evidence(file_name=file_name, page=page, quote=quote),),
        confidence=0.9,
    )


def derived(amount: str = "400000", **overrides: object) -> Derivation:
    base: dict[str, Any] = {
        "requirement_id": "req-2",
        "fact_kind": "capital_expenditure",
        "unit": Unit.MONEY,
        "currency": Currency.USD,
        "identity": "additions = closing − opening + depreciation + disposals",
        "terms": (
            DerivationTerm(
                label="closing",
                sign=Sign.PLUS,
                amount=Decimal("1450000"),
                evidence=Evidence(file_name="notes.pdf", page=2, quote=MOVEMENT),
            ),
            DerivationTerm(
                label="opening",
                sign=Sign.MINUS,
                amount=Decimal("1200000"),
                evidence=Evidence(file_name="notes.pdf", page=2, quote=MOVEMENT),
            ),
            DerivationTerm(
                label="depreciation",
                sign=Sign.PLUS,
                amount=Decimal("150000"),
                evidence=Evidence(file_name="notes.pdf", page=2, quote=MOVEMENT),
            ),
        ),
        "amount": Decimal(amount),
        "confidence": 0.8,
    }
    return Derivation(**(base | overrides))


def proposed(**overrides: object) -> ProposedAdjustment:
    base: dict[str, Any] = {
        "action": AdjustmentAction.RECLASSIFY,
        "reason": "статья переклассифицирована примечанием 4",
        "evidence": Evidence(file_name="notes.pdf", page=1, quote=FLOOR),
        "txn_id": "P1-000123",
        "new_value": "capex",
    }
    return ProposedAdjustment(**(base | overrides))


def answer(
    facts: tuple[ResolvedFact, ...] = (),
    derivations: tuple[Derivation, ...] = (),
    adjustments: tuple[ProposedAdjustment, ...] = (),
) -> dict[str, Any]:
    return ResolverResponse(
        facts=facts, derivations=derivations, adjustments=adjustments
    ).model_dump(mode="json")


class Responder:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.sent = 0

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self.sent += 1
        return self.outcomes[min(self.sent - 1, len(self.outcomes) - 1)], (100, 20), None


def resolver(tmp_path: Path, responder: Responder) -> RequirementResolver:
    return RequirementResolver(
        runner=StructuredModelRunner(
            config=ModelConfig(name="test-model", api_key="sk-test"),
            cache=ModelCache(directory=tmp_path, policy=CachePolicy.READ_WRITE),
            budget=Budget(),
            send=responder,
        )
    )


def batch(
    *requirements: FactRequirement, documents: tuple[DocumentFacts, ...] = ()
) -> ResolverBatch:
    return ResolverBatch.build(ACCOUNT, requirements or (need(),), documents or (document(),))


# --- валидный ответ -----------------------------------------------------------


def test_open_requirement_is_read_from_the_notes(tmp_path: Path) -> None:
    responder = Responder(answer(facts=(found(),)))
    result = resolver(tmp_path, responder).resolve(batch())

    assert result.answered == ("req-1",)
    assert result.facts[0].amount == Decimal("50000")
    assert result.unresolved == ()
    assert responder.sent == 1


def test_derivation_is_added_up_by_the_code(tmp_path: Path) -> None:
    """Число, которого в отчётности нет строкой, собирается из раскрытых слагаемых."""
    responder = Responder(answer(derivations=(derived(),)))
    result = resolver(tmp_path, responder).resolve(batch(need("req-2", kind="capital_expenditure")))

    assert result.derivations[0].total() == Decimal("400000")
    assert result.answered == ("req-2",)


def test_adjustment_gets_its_status_from_the_code(tmp_path: Path) -> None:
    """Статус модель не называет: он выводится из права документа менять расчёт."""
    responder = Responder(answer(facts=(found(),), adjustments=(proposed(),)))
    result = resolver(tmp_path, responder).resolve(batch())

    adjustment = result.adjustments[0]
    assert isinstance(adjustment, ReclassifyAdjustment)
    assert adjustment.status is AdjustmentStatus.PENDING
    assert adjustment.account_id == ACCOUNT
    assert adjustment.source.file_hash == document().sha256


def test_adjustment_from_a_draft_is_recorded_but_unconfirmed(tmp_path: Path) -> None:
    """Ведомость реестр не меняет, но «не подтверждено» и «не найдено» — разное."""
    draft = document(kind=DocumentKind.AUDIT_PROCEDURES, status=DocumentStatus.DRAFT)
    responder = Responder(answer(adjustments=(proposed(),)))
    result = resolver(tmp_path, responder).resolve(batch(documents=(draft,)))

    assert result.adjustments[0].status is AdjustmentStatus.UNCONFIRMED


def test_second_run_reads_the_cache(tmp_path: Path) -> None:
    responder = Responder(answer(facts=(found(),)))
    engine = resolver(tmp_path, responder)

    engine.resolve(batch())
    engine.resolve(batch())
    assert responder.sent == 1


def test_empty_batch_costs_nothing(tmp_path: Path) -> None:
    """Требований нет — платить за запрос «найди ничего» незачем."""
    responder = Responder(answer())
    closed = need(resolution=Resolution.RESOLVED)
    result = resolver(tmp_path, responder).resolve(batch(closed))

    assert result.answered == ()
    assert responder.sent == 0


# --- частичный ответ ----------------------------------------------------------


def test_unanswered_requirement_stays_open(tmp_path: Path) -> None:
    """Ненайденная величина не ошибка ответа: она честно доедет до отчёта."""
    responder = Responder(answer(facts=(found("req-1"),)))
    result = resolver(tmp_path, responder).resolve(
        batch(need("req-1"), need("req-2", kind="collateral_coverage"))
    )

    assert result.answered == ("req-1",)
    assert result.unresolved == ("req-2",)
    assert responder.sent == 1


def test_empty_answer_is_accepted(tmp_path: Path) -> None:
    responder = Responder(answer())
    result = resolver(tmp_path, responder).resolve(batch())

    assert result.unresolved == ("req-1",)
    assert responder.sent == 1


# --- неоднозначный ответ ------------------------------------------------------


def test_two_answers_to_one_requirement_resolve_nothing(tmp_path: Path) -> None:
    """Выбор одного из двух зависел бы от порядка чтения, поэтому не выбираем."""
    responder = Responder(answer(facts=(found(amount="50000"), found(amount="75000"))))
    result = resolver(tmp_path, responder).resolve(batch())

    assert result.ambiguous == ("req-1",)
    assert result.facts == ()
    assert result.unresolved == ("req-1",)


def test_ambiguity_counts_derivations_too(tmp_path: Path) -> None:
    responder = Responder(
        answer(facts=(found("req-2", kind="capital_expenditure"),), derivations=(derived(),))
    )
    result = resolver(tmp_path, responder).resolve(batch(need("req-2", kind="capital_expenditure")))

    assert result.ambiguous == ("req-2",)
    assert result.derivations == ()


def test_requirement_awaiting_many_facts_accepts_several(tmp_path: Path) -> None:
    """Разовых статей у заёмщика может быть несколько, и это не спор."""
    responder = Responder(
        answer(
            facts=(
                found(kind="one_off_item", amount="50000"),
                found(kind="one_off_item", amount="75000"),
            )
        )
    )
    result = resolver(tmp_path, responder).resolve(
        batch(need(kind="one_off_item", cardinality=Cardinality.MANY))
    )

    assert result.ambiguous == ()
    assert len(result.facts) == 2


# --- невалидный ответ ---------------------------------------------------------


def test_paraphrased_quote_is_rejected(tmp_path: Path) -> None:
    responder = Responder(answer(facts=(found(quote="разовой считается статья от 50 тысяч"),)))
    engine = resolver(tmp_path, responder)

    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert responder.sent == 2
    assert "quote_is_not_found" in engine.runner.calls[0].note


def test_quote_from_another_page_is_rejected(tmp_path: Path) -> None:
    responder = Responder(answer(facts=(found(page=2),)))
    with pytest.raises(InvalidResponseError):
        resolver(tmp_path, responder).resolve(batch())


def test_unknown_requirement_is_rejected(tmp_path: Path) -> None:
    engine = resolver(tmp_path, Responder(answer(facts=(found("req-99"),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert "requirement_is_unknown" in engine.runner.calls[0].note


def test_substituted_fact_kind_is_rejected(tmp_path: Path) -> None:
    """Требование про порог существенности не закрывается суммой обязательства."""
    engine = resolver(tmp_path, Responder(answer(facts=(found(kind="aggregate_obligation"),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert "fact_kind_mismatch" in engine.runner.calls[0].note


def test_wrong_unit_is_rejected(tmp_path: Path) -> None:
    engine = resolver(tmp_path, Responder(answer(facts=(found(unit=Unit.RATIO, currency=None),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert "unit_mismatch" in engine.runner.calls[0].note


def test_wrong_currency_is_rejected(tmp_path: Path) -> None:
    engine = resolver(tmp_path, Responder(answer(facts=(found(currency=Currency.KZT),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert "currency_mismatch" in engine.runner.calls[0].note


def test_derivation_that_does_not_add_up_is_rejected(tmp_path: Path) -> None:
    """Итог модели разошёлся с её же слагаемыми — верить нельзя ни тому, ни другому."""
    engine = resolver(tmp_path, Responder(answer(derivations=(derived(amount="500000"),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch(need("req-2", kind="capital_expenditure")))
    assert "derivation_does_not_add_up" in engine.runner.calls[0].note


def test_adjustment_without_its_required_field_is_rejected(tmp_path: Path) -> None:
    engine = resolver(tmp_path, Responder(answer(adjustments=(proposed(new_value=None),))))
    with pytest.raises(InvalidResponseError):
        engine.resolve(batch())
    assert "adjustment_is_incomplete" in engine.runner.calls[0].note


def test_value_from_an_ignored_document_is_rejected(tmp_path: Path) -> None:
    """Вытесненная редакция читается и хранится, но величину из неё мы не берём."""
    old = document(status=DocumentStatus.SUPERSEDED)
    engine = resolver(tmp_path, Responder(answer(facts=(found(),))))

    with pytest.raises(InvalidResponseError):
        engine.resolve(batch(documents=(old,)))
    assert "source_is_ignored" in engine.runner.calls[0].note


def test_value_from_a_draft_is_rejected_but_its_adjustment_is_not(tmp_path: Path) -> None:
    draft = document(kind=DocumentKind.AUDIT_PROCEDURES, status=DocumentStatus.DRAFT)
    documents = {draft.file_name: draft}
    requirements = [need()]

    value = validate(
        ResolverResponse(facts=(found(),)), requirements=requirements, documents=documents
    )
    change = validate(
        ResolverResponse(adjustments=(proposed(),)), requirements=requirements, documents=documents
    )
    assert [error.code for error in value] == ["source_is_not_authoritative"]
    assert change == []


def test_invalid_answer_is_not_cached(tmp_path: Path) -> None:
    responder = Responder(answer(facts=(found(quote="пересказ"),)))
    with pytest.raises(InvalidResponseError):
        resolver(tmp_path, responder).resolve(batch())
    assert list(tmp_path.glob("*.json")) == []


def test_bad_answer_costs_one_semantic_retry(tmp_path: Path) -> None:
    """Повтор делает Runner: собственных вторых запросов у resolver нет."""
    responder = Responder(answer(facts=(found(quote="пересказ"),)), answer(facts=(found(),)))
    engine = resolver(tmp_path, responder)
    result = engine.resolve(batch())

    assert result.answered == ("req-1",)
    assert responder.sent == 2
    assert [call.attempt for call in engine.runner.calls] == [1, 2]
    assert engine.runner.usage()["invalid_responses"] == 1


# --- изоляция и детерминированность -------------------------------------------


def test_foreign_document_stops_the_batch() -> None:
    with pytest.raises(BorrowerIsolationError, match=r"other\.pdf"):
        ResolverBatch.build(
            ACCOUNT, [need()], [document(), document("other.pdf", account="ACC-9999")]
        )


def test_requirements_of_another_borrower_do_not_enter_the_batch() -> None:
    mine = need("req-1")
    alien = need("req-2", account_id="ACC-9999")
    assert ResolverBatch.build(ACCOUNT, [mine, alien], [document()]).requirements == (mine,)


def test_resolved_requirements_do_not_enter_the_batch() -> None:
    """Найденное детерминированным слоем модели не показывают: она его переспорит."""
    closed = need("req-1", resolution=Resolution.RESOLVED)
    open_one = need("req-2", kind="collateral_coverage")
    batch = ResolverBatch.build(ACCOUNT, [closed, open_one], [document()])
    assert batch.requirements == (open_one,)


def test_input_order_does_not_change_the_request(tmp_path: Path) -> None:
    first, second = need("req-1"), need("req-2", kind="collateral_coverage")
    left, right = document("a.pdf"), document("b.pdf")
    engine = resolver(tmp_path, Responder(answer()))

    straight = ResolverBatch.build(ACCOUNT, [first, second], [left, right])
    reversed_ = ResolverBatch.build(ACCOUNT, [second, first], [right, left])
    assert straight.payload() == reversed_.payload()
    assert engine.runner.key(engine.request(straight), attempt=1) == engine.runner.key(
        engine.request(reversed_), attempt=1
    )


def test_request_is_marked_as_the_resolver(tmp_path: Path) -> None:
    """Роль входит в ключ кэша: ответ компилятора не должен подойти resolver."""
    engine = resolver(tmp_path, Responder(answer()))
    assert engine.request(batch()).role is Role.RESOLVER


# --- схема запроса ------------------------------------------------------------


def test_schema_is_strict() -> None:
    schema = strict_schema(ResolverResponse)
    assert set(schema["required"]) == {"facts", "derivations", "adjustments"}
    assert schema["additionalProperties"] is False
    assert "'default'" not in repr(schema)


def test_schema_has_no_place_for_a_verdict() -> None:
    """Ни статуса, ни ячейки Submission: вердикт resolver не выносит."""
    text = repr(strict_schema(ResolverResponse))
    for forbidden in ("status", "verdict", "actual", "evidence_transaction_id"):
        assert f"'{forbidden}'" not in text
