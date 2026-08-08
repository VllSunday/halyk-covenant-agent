"""Превращение того, что дочитал resolver, в факт, пригодный для расчёта.

Ответ модели и факт — разные вещи. Ответ адресуется требованию компилятора и несёт
цитату; факт адресуется заёмщику и несёт сумму в валюте, с которой умеет работать
исполнитель. Между ними нужен явный переход, и он здесь.

Переход неполный, и это осознанно. Исполнитель читает факты трёх видов: разовую
статью, порог существенности и совокупное обязательство. Величина, которую он в
дерево подставить не сможет, отклоняется здесь с названной причиной, а не доезжает
до расчёта, чтобы там превратиться в «факта не нашлось»: первое чинится правкой
контракта, второе — поиском несуществующего документа.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from halyk.compiler.contract import FactRequirement
from halyk.models.covenant import Unit
from halyk.models.document import DocumentFacts
from halyk.models.fact import (
    AggregateObligationFact,
    Fact,
    FactKind,
    OneOffItemFact,
    OneOffPolicyFact,
)
from halyk.models.source import SourceRef
from halyk.money import Currency, Money, MoneyParseError
from halyk.resolution.batch import ResolutionResult, source_ref
from halyk.resolution.contract import Evidence

# Виды, которые исполнитель умеет подставить в дерево. Список закрыт: расширять его
# без правки исполнителя бессмысленно — факт просто не встретится ни одному узлу.
EXECUTABLE_KINDS = frozenset(
    {FactKind.ONE_OFF_ITEM, FactKind.ONE_OFF_POLICY, FactKind.AGGREGATE_OBLIGATION}
)


@dataclass(frozen=True, slots=True)
class BridgeError:
    """Дочитанная величина, которую нельзя подставить в расчёт."""

    requirement_id: str
    code: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"requirement_id": self.requirement_id, "code": self.code, "detail": self.detail}


def _money(amount: Decimal, requirement: FactRequirement, currency: Currency | None) -> Money:
    resolved = currency or requirement.currency
    if resolved is None:
        raise MoneyParseError("валюта не названа ни в ответе, ни в требовании")
    return Money.from_decimal(amount, resolved)


def _fact(
    requirement: FactRequirement, amount: Money, source: SourceRef, account_id: str
) -> Fact | None:
    """Факт того вида, который объявило требование. Описание берётся из него же.

    Описание не спрашивается у модели намеренно: по нему исполнитель отбирает статьи
    внутри вида (`description_contains`), и придуманная формулировка молча увела бы
    величину мимо узла, который её ждёт.
    """
    common = {"account_id": account_id, "source": source}
    match requirement.fact_kind:
        case FactKind.ONE_OFF_POLICY:
            return OneOffPolicyFact(minimum=amount, **common)
        case FactKind.ONE_OFF_ITEM:
            return OneOffItemFact(
                description=requirement.description,
                counterparty=requirement.counterparty or "",
                amount=amount,
                **common,
            )
        case FactKind.AGGREGATE_OBLIGATION:
            return AggregateObligationFact(
                description=requirement.description, amount=amount, **common
            )
    return None


@dataclass(frozen=True, slots=True)
class _Answer:
    """Ответ модели в форме, одинаковой для найденной и для выведенной величины."""

    requirement_id: str
    amount: Decimal
    unit: Unit
    currency: Currency | None
    evidence: Evidence


def _build(
    answer: _Answer,
    requirement: FactRequirement | None,
    documents: Mapping[str, DocumentFacts],
    account_id: str,
) -> tuple[Fact | None, BridgeError | None]:
    name = answer.requirement_id
    if requirement is None:
        return None, BridgeError(
            requirement_id=name,
            code="requirement_is_unknown",
            detail="ответ ссылается на требование, которого нет среди открытых",
        )
    if requirement.fact_kind not in EXECUTABLE_KINDS:
        return None, BridgeError(
            requirement_id=name,
            code="fact_kind_is_not_executable",
            detail=f"величину вида {requirement.fact_kind} исполнитель подставить не умеет",
        )
    if answer.unit is not Unit.MONEY:
        return None, BridgeError(
            requirement_id=name,
            code="unit_is_not_money",
            detail=f"величина в {answer.unit.value}: факт хранит только денежную сумму",
        )
    try:
        money = _money(answer.amount, requirement, answer.currency)
        source = source_ref(answer.evidence.file_name, answer.evidence.page, dict(documents))
    except (MoneyParseError, ValueError) as exc:
        return None, BridgeError(requirement_id=name, code="fact_is_not_built", detail=str(exc))
    return _fact(requirement, money, source, account_id), None


def to_facts(
    result: ResolutionResult,
    requirements: Sequence[FactRequirement],
    documents: Mapping[str, DocumentFacts],
) -> tuple[tuple[Fact, ...], tuple[BridgeError, ...]]:
    """Факты и отказы по одному заёмщику.

    Выведенная величина приходит слагаемыми, и в расчёт идёт их сумма, а не итог,
    названный моделью: сходимость этих двух чисел проверил валидатор, и брать после
    него объявленный итог значило бы держать проверку ради отчёта.
    """
    known = {item.requirement_id: item for item in requirements}
    facts: list[Fact] = []
    errors: list[BridgeError] = []

    answers = [
        _Answer(item.requirement_id, item.amount, item.unit, item.currency, item.evidence[0])
        for item in result.facts
    ]
    answers += [
        _Answer(item.requirement_id, item.total(), item.unit, item.currency, item.terms[0].evidence)
        for item in result.derivations
    ]

    for answer in answers:
        fact, error = _build(answer, known.get(answer.requirement_id), documents, result.account_id)
        if error is not None:
            errors.append(error)
        elif fact is not None:
            facts.append(fact)
    return tuple(facts), tuple(errors)


__all__ = ["EXECUTABLE_KINDS", "BridgeError", "to_facts"]
