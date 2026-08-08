"""Проверка того, что вернул resolver, до попадания величины в расчёт.

Проверяется всё, что сверяется с документом или с требованием: адрес и дословность
цитаты, право документа менять расчёт, совпадение вида и размерности величины,
сходимость выведенного числа. Ответ, который нельзя сверить, к расчёту не
допускается — так же, как у компилятора.

Право документа спрашивается по-разному у величины и у корректировки, и это не
недосмотр. Величина уходит прямо в формулу, поэтому источником ей может быть только
действующий и окончательный документ. Корректировка же и из промежуточной ведомости
попадает в отчёт — со статусом «не подтверждена»: половина ловушек датасета в том,
что документ рассмотрел вопрос и ничего не поменял, и отличить это от «мы не
заметили» можно только записью.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from halyk.compiler.contract import FactRequirement
from halyk.compiler.dimensions import dimension_of
from halyk.compiler.validator import contains_quote
from halyk.knowledge.authority import resolve_authority
from halyk.models.covenant import Unit
from halyk.models.document import DocumentFacts
from halyk.models.source import SourceAuthority
from halyk.money import Currency
from halyk.resolution.contract import (
    Derivation,
    Evidence,
    ProposedAdjustment,
    ResolvedFact,
    ResolverResponse,
)


@dataclass(frozen=True, slots=True)
class ResolutionError:
    """Отклонённая часть ответа с адресом внутри него."""

    code: str
    path: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


def check_quote(
    evidence: Evidence, documents: Mapping[str, DocumentFacts], path: str
) -> list[ResolutionError]:
    """Цитата обязана дословно стоять на заявленной странице заявленного файла."""
    document = documents.get(evidence.file_name)
    if document is None:
        return [
            ResolutionError(
                code="source_is_unknown",
                path=path,
                detail=f"файла {evidence.file_name} нет среди документов заёмщика",
            )
        ]
    pages = [page for page in document.pages if page.number == evidence.page]
    if not pages:
        return [
            ResolutionError(
                code="page_is_missing",
                path=path,
                detail=f"в {evidence.file_name} нет страницы {evidence.page}",
            )
        ]
    if not contains_quote(pages[0].text, evidence.quote):
        return [
            ResolutionError(
                code="quote_is_not_found",
                path=path,
                detail=f"цитаты нет на странице {evidence.page} файла {evidence.file_name}",
            )
        ]
    return []


def check_authority(
    evidence: Evidence,
    documents: Mapping[str, DocumentFacts],
    path: str,
    *,
    required: SourceAuthority,
) -> list[ResolutionError]:
    """Документ, на который ссылается ответ, вправе влиять на расчёт."""
    document = documents.get(evidence.file_name)
    if document is None:
        return []

    authority = resolve_authority(document)
    if authority is SourceAuthority.IGNORED:
        return [
            ResolutionError(
                code="source_is_ignored",
                path=path,
                detail=f"{evidence.file_name}: посторонний или вытесненный документ",
            )
        ]
    if required is SourceAuthority.AUTHORITATIVE and authority is not required:
        return [
            ResolutionError(
                code="source_is_not_authoritative",
                path=path,
                detail=f"{evidence.file_name}: {authority.value}, величину из него брать нельзя",
            )
        ]
    return []


def check_requirement(
    requirement: FactRequirement | None, fact_kind: str, requirement_id: str, path: str
) -> list[ResolutionError]:
    """Ответ ссылается на открытое требование этого заёмщика и не подменяет его вид."""
    if requirement is None:
        return [
            ResolutionError(
                code="requirement_is_unknown",
                path=path,
                detail=f"{requirement_id} нет среди открытых требований батча",
            )
        ]
    if requirement.fact_kind != fact_kind:
        return [
            ResolutionError(
                code="fact_kind_mismatch",
                path=path,
                detail=f"требование {requirement_id} про {requirement.fact_kind}, а не {fact_kind}",
            )
        ]
    return []


def check_dimensions(
    requirement: FactRequirement, unit: Unit, currency: Currency | None, path: str
) -> list[ResolutionError]:
    """Размерность и валюта найденной величины совпадают с объявленными в требовании."""
    found: list[ResolutionError] = []
    if dimension_of(unit) is not dimension_of(requirement.unit):
        found.append(
            ResolutionError(
                code="unit_mismatch",
                path=path,
                detail=f"требование объявлено в {requirement.unit.value}, ответ в {unit.value}",
            )
        )
    wanted = requirement.currency
    if unit is Unit.MONEY and currency is None:
        found.append(
            ResolutionError(
                code="currency_is_missing", path=path, detail="денежная величина без валюты"
            )
        )
    elif wanted is not None and currency is not None and currency != wanted:
        found.append(
            ResolutionError(
                code="currency_mismatch",
                path=path,
                detail=f"требование в {wanted}, ответ в {currency}",
            )
        )
    return found


def check_derivation(derivation: Derivation, path: str) -> list[ResolutionError]:
    """Итог, названный моделью, обязан сойтись с суммой её же слагаемых."""
    if derivation.total() == derivation.amount:
        return []
    return [
        ResolutionError(
            code="derivation_does_not_add_up",
            path=path,
            detail=f"слагаемые дают {derivation.total()}, объявлено {derivation.amount}",
        )
    ]


def _evidence_errors(
    items: Sequence[Evidence],
    documents: Mapping[str, DocumentFacts],
    path: str,
    *,
    required: SourceAuthority,
) -> list[ResolutionError]:
    found: list[ResolutionError] = []
    for position, evidence in enumerate(items):
        where = f"{path}.evidence[{position}]"
        found += check_quote(evidence, documents, where)
        found += check_authority(evidence, documents, where, required=required)
    return found


def validate_fact(
    fact: ResolvedFact,
    requirements: Mapping[str, FactRequirement],
    documents: Mapping[str, DocumentFacts],
    path: str,
) -> list[ResolutionError]:
    requirement = requirements.get(fact.requirement_id)
    found = check_requirement(requirement, fact.fact_kind, fact.requirement_id, path)
    if requirement is not None:
        found += check_dimensions(requirement, fact.unit, fact.currency, path)
    found += _evidence_errors(
        fact.evidence, documents, path, required=SourceAuthority.AUTHORITATIVE
    )
    return found


def validate_derivation(
    derivation: Derivation,
    requirements: Mapping[str, FactRequirement],
    documents: Mapping[str, DocumentFacts],
    path: str,
) -> list[ResolutionError]:
    requirement = requirements.get(derivation.requirement_id)
    found = check_requirement(requirement, derivation.fact_kind, derivation.requirement_id, path)
    if requirement is not None:
        found += check_dimensions(requirement, derivation.unit, derivation.currency, path)
    found += check_derivation(derivation, path)
    found += _evidence_errors(
        [term.evidence for term in derivation.terms],
        documents,
        path,
        required=SourceAuthority.AUTHORITATIVE,
    )
    return found


def validate_adjustment(
    adjustment: ProposedAdjustment, documents: Mapping[str, DocumentFacts], path: str
) -> list[ResolutionError]:
    return _evidence_errors(
        [adjustment.evidence], documents, path, required=SourceAuthority.RECORDED
    )


def validate(
    response: ResolverResponse,
    *,
    requirements: Iterable[FactRequirement],
    documents: Mapping[str, DocumentFacts],
) -> list[ResolutionError]:
    """Все проверки ответа. Неоднозначности здесь нет — она не ошибка ответа."""
    known = {item.requirement_id: item for item in requirements}
    found: list[ResolutionError] = []
    for position, fact in enumerate(response.facts):
        found += validate_fact(fact, known, documents, f"facts[{position}]")
    for position, derivation in enumerate(response.derivations):
        found += validate_derivation(derivation, known, documents, f"derivations[{position}]")
    for position, adjustment in enumerate(response.adjustments):
        found += validate_adjustment(adjustment, documents, f"adjustments[{position}]")
    return found
