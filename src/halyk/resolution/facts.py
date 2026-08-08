"""Превращение того, что дочитал resolver, в факт, пригодный для расчёта.

Ответ модели и факт — разные вещи. Ответ адресуется требованию компилятора и несёт
цитату; факт адресуется заёмщику и несёт величину в той форме, с которой работает
исполнитель. Между ними нужен явный переход, и он здесь.

Специализированные виды разбираются первыми: разовая статья, порог существенности и
совокупное обязательство ложатся в свои модели, потому что исполнитель уже умеет
отбирать их по своим правилам. Всё остальное становится `ResolvedMetricFact` — вид
величины назначает компилятор, и новый показатель не должен требовать новой модели
данных.

Отклонить величину здесь можно только по причине, которую видно в отчёте: требование
не из этого батча, адрес не из переписи, денежная сумма без валюты. Всё остальное —
цитаты, право документа, размерности, покрытие требований — проверено раньше, и
повторять эти проверки значило бы держать два описания одного контракта.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from halyk.compiler.contract import FactRequirement
from halyk.models.covenant import Unit
from halyk.models.document import DocumentFacts
from halyk.models.fact import (
    AggregateObligationFact,
    DerivedTerm,
    Fact,
    FactKind,
    OneOffItemFact,
    OneOffPolicyFact,
    Qualifier,
    ResolvedMetricFact,
)
from halyk.models.source import SourceRef
from halyk.money import Currency, Money, MoneyParseError
from halyk.resolution.batch import ResolutionResult, source_ref
from halyk.resolution.contract import Evidence

# Виды, у которых есть собственная модель факта и собственные правила отбора в
# исполнителе. Всё, чего здесь нет, доезжает до расчёта общей величиной.
SPECIALISED_KINDS = frozenset(
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


@dataclass(frozen=True, slots=True)
class _Term:
    """Слагаемое выведенной величины до того, как у него появился адрес."""

    label: str
    amount: Decimal
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class _Answer:
    """Ответ модели в форме, одинаковой для найденной и для выведенной величины."""

    requirement_id: str
    amount: Decimal
    unit: Unit
    currency: Currency | None
    evidence: Evidence
    confidence: float = 1.0
    derivation: str = ""
    terms: tuple[_Term, ...] = ()
    supporting: tuple[Evidence, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class _Located:
    """Адреса ответа, сверенные с переписью."""

    source: SourceRef
    supporting: tuple[SourceRef, ...]
    terms: tuple[DerivedTerm, ...]


def _locate(answer: _Answer, documents: Mapping[str, DocumentFacts]) -> _Located:
    known = dict(documents)
    return _Located(
        source=source_ref(answer.evidence.file_name, answer.evidence.page, known),
        supporting=tuple(
            source_ref(item.file_name, item.page, known) for item in answer.supporting
        ),
        terms=tuple(
            DerivedTerm(
                label=term.label,
                amount=term.amount,
                source=source_ref(term.evidence.file_name, term.evidence.page, known),
            )
            for term in answer.terms
        ),
    )


def _specialised(
    requirement: FactRequirement, amount: Money, source: SourceRef, account_id: str
) -> Fact | None:
    """Факт того вида, который объявило требование, если такой вид у нас есть.

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


def _generic(
    answer: _Answer, requirement: FactRequirement, located: _Located, account_id: str
) -> ResolvedMetricFact:
    """Величина под требование, для которого своей модели факта нет.

    Всё, что позволяет проверить её задним числом, хранится рядом: требование, вид,
    период, область, уточнения отбора и адреса, по которым она прочитана.
    """
    return ResolvedMetricFact(
        account_id=account_id,
        source=located.source,
        requirement_id=requirement.requirement_id,
        scenario_id=requirement.scenario_id,
        metric=requirement.fact_kind,
        description=requirement.description,
        value=answer.amount,
        unit=answer.unit,
        currency=answer.currency or requirement.currency,
        period_start=requirement.period_start,
        period_end=requirement.period_end,
        scope=requirement.scope,
        qualifiers=(
            (Qualifier(name="counterparty", value=requirement.counterparty),)
            if requirement.counterparty
            else ()
        ),
        derivation=answer.derivation,
        terms=located.terms,
        supporting=located.supporting,
        confidence=answer.confidence,
    )


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
    try:
        located = _locate(answer, documents)
        if requirement.fact_kind not in SPECIALISED_KINDS:
            return _generic(answer, requirement, located, account_id), None

        currency = answer.currency or requirement.currency
        if answer.unit is not Unit.MONEY or currency is None:
            raise MoneyParseError(
                f"величина вида {requirement.fact_kind} хранится суммой, "
                f"а пришла как {answer.unit.value} без валюты"
            )
        money = Money.from_decimal(answer.amount, currency)
    except (MoneyParseError, ValueError) as exc:
        return None, BridgeError(requirement_id=name, code="fact_is_not_built", detail=str(exc))
    return _specialised(requirement, money, located.source, account_id), None


def _answers(result: ResolutionResult) -> list[_Answer]:
    """Найденные и выведенные величины одним списком.

    Выведенная приходит слагаемыми, и в расчёт идёт их сумма, а не итог, названный
    моделью: сходимость этих двух чисел проверил валидатор, и брать после него
    объявленный итог значило бы держать проверку ради отчёта.
    """
    found = [
        _Answer(
            requirement_id=item.requirement_id,
            amount=item.amount,
            unit=item.unit,
            currency=item.currency,
            evidence=item.evidence[0],
            confidence=item.confidence,
            supporting=tuple(item.evidence[1:]),
        )
        for item in result.facts
    ]
    return found + [
        _Answer(
            requirement_id=item.requirement_id,
            amount=item.total(),
            unit=item.unit,
            currency=item.currency,
            evidence=item.terms[0].evidence,
            confidence=item.confidence,
            derivation=item.identity,
            terms=tuple(
                _Term(
                    label=term.label,
                    amount=term.sign.factor * term.amount,
                    evidence=term.evidence,
                )
                for term in item.terms
            ),
        )
        for item in result.derivations
    ]


def to_facts(
    result: ResolutionResult,
    requirements: Sequence[FactRequirement],
    documents: Mapping[str, DocumentFacts],
) -> tuple[tuple[Fact, ...], tuple[BridgeError, ...]]:
    """Факты и отказы по одному заёмщику."""
    known = {item.requirement_id: item for item in requirements}
    facts: list[Fact] = []
    errors: list[BridgeError] = []

    for answer in _answers(result):
        fact, error = _build(answer, known.get(answer.requirement_id), documents, result.account_id)
        if error is not None:
            errors.append(error)
        elif fact is not None:
            facts.append(fact)
    return tuple(facts), tuple(errors)


__all__ = ["SPECIALISED_KINDS", "BridgeError", "to_facts"]
