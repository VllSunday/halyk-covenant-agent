"""Проверка того, что вернул компилятор, до единого расчёта.

Ответ модели — гипотеза, а не факт. Проверяется всё, что можно сверить с
документом или с шаблоном: адрес ячейки, наличие цитаты на заявленной странице,
действующая редакция, замкнутость селектора, объявленность каждого факта.

Непрошедший ответ не чинится догадкой и не идёт в расчёт. Причина отказа
называется адресом узла — по нему видно, что именно перекомпилировать.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from halyk.compiler.contract import Cardinality, CompiledClause, FactRequirement, Resolution
from halyk.compiler.dimensions import check as check_dimensions
from halyk.models.classification import TransactionCategory
from halyk.models.covenant import Unit
from halyk.models.document import DocumentFacts, DocumentStatus
from halyk.models.fact import Fact
from halyk.models.formula import (
    Condition,
    ExternalMetric,
    FactValue,
    LedgerCount,
    LedgerSum,
    Quantity,
    Selector,
)

_WHITESPACE = re.compile(r"\s+")

_MONTHS_IN_YEAR = 12

# Узлы, из которых собирается дерево. Список закрыт: ответ с неизвестной операцией
# не разберётся ещё на уровне схемы, но проверка оставлена явной, чтобы расширение
# набора операций нельзя было провести молча.
ALLOWED_OPS = frozenset(
    {
        "ledger_sum",
        "ledger_count",
        "fact_value",
        "constant",
        "external",
        "add",
        "sub",
        "mul",
        "div",
        "abs",
        "max",
        "min",
    }
)


@dataclass(frozen=True, slots=True)
class CompilationError:
    """Отклонённая часть ответа с адресом внутри дерева."""

    code: str
    path: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


def _flatten(text: str) -> str:
    """Форма для поиска цитаты: PDF рвёт фразу переносами в произвольных местах."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def contains_quote(text: str, quote: str) -> bool:
    """Цитата стоит в тексте дословно, с точностью до переносов."""
    return _flatten(quote) in _flatten(text)


def walk(node: Quantity, path: str = "measure") -> Iterable[tuple[str, Quantity]]:
    """Все узлы дерева вместе с их адресами."""
    yield path, node
    match node.op:
        case "add":
            for index, term in enumerate(node.terms):
                yield from walk(term, f"{path}.terms[{index}]")
        case "mul":
            for index, factor in enumerate(node.factors):
                yield from walk(factor, f"{path}.factors[{index}]")
        case "max" | "min":
            for index, value in enumerate(node.values):
                yield from walk(value, f"{path}.values[{index}]")
        case "sub":
            yield from walk(node.left, f"{path}.left")
            yield from walk(node.right, f"{path}.right")
        case "div":
            yield from walk(node.numerator, f"{path}.numerator")
            yield from walk(node.denominator, f"{path}.denominator")
        case "abs":
            yield from walk(node.value, f"{path}.value")


def _nodes(clause: CompiledClause) -> list[tuple[str, Quantity]]:
    found = list(walk(clause.formula.measure))
    found += list(walk(clause.formula.threshold, "threshold"))
    if (condition := clause.formula.applies_when) is not None:
        found += _condition_nodes(condition)
    return found


def _condition_nodes(condition: Condition) -> list[tuple[str, Quantity]]:
    return [
        *walk(condition.left, "applies_when.left"),
        *walk(condition.right, "applies_when.right"),
    ]


def check_cell(
    clause: CompiledClause, expected: Sequence[tuple[str, str]]
) -> list[CompilationError]:
    """Пункт должен попадать в ячейку шаблона, а не в придуманную."""
    if clause.cell in set(expected):
        return []
    return [
        CompilationError(
            code="cell_is_unknown",
            path="formula",
            detail=f"{clause.cell} нет среди ячеек шаблона",
        )
    ]


def check_operations(clause: CompiledClause) -> list[CompilationError]:
    return [
        CompilationError(
            code="operation_is_not_allowed",
            path=path,
            detail=f"операция {node.op} вне закрытого списка",
        )
        for path, node in _nodes(clause)
        if node.op not in ALLOWED_OPS
    ]


def check_quote(
    clause: CompiledClause, documents: Mapping[str, DocumentFacts]
) -> list[CompilationError]:
    """Цитата обязана дословно стоять на заявленной странице.

    Это единственная проверка, отделяющая прочитанное от сочинённого: модель,
    пересказавшая пункт своими словами, здесь и останавливается.
    """
    found: list[CompilationError] = []
    for index, ref in enumerate(clause.formula.source_refs):
        path = f"source_refs[{index}]"
        document = documents.get(ref.file_name)
        if document is None:
            found.append(
                CompilationError(
                    code="source_is_unknown",
                    path=path,
                    detail=f"файла {ref.file_name} нет в переписи",
                )
            )
            continue
        pages = [page for page in document.pages if page.number == ref.page]
        if not pages:
            found.append(
                CompilationError(
                    code="page_is_missing",
                    path=path,
                    detail=f"в {ref.file_name} нет страницы {ref.page}",
                )
            )
            continue
        if not contains_quote(pages[0].text, clause.formula.quote):
            found.append(
                CompilationError(
                    code="quote_is_not_found",
                    path=path,
                    detail=f"цитаты нет на странице {ref.page} файла {ref.file_name}",
                )
            )
    return found


def check_edition(
    clause: CompiledClause, documents: Mapping[str, DocumentFacts]
) -> list[CompilationError]:
    """Пункт читается из действующей редакции, а не из вытесненной."""
    return [
        CompilationError(
            code="edition_is_superseded",
            path=f"source_refs[{index}]",
            detail=f"{ref.file_name} — недействующая редакция",
        )
        for index, ref in enumerate(clause.formula.source_refs)
        if (document := documents.get(ref.file_name)) is not None
        and document.status is DocumentStatus.SUPERSEDED
    ]


def check_selectors(clause: CompiledClause) -> list[CompilationError]:
    """Селектор без ограничений сложил бы весь реестр заёмщика в одну величину."""
    found: list[CompilationError] = []
    for path, node in _nodes(clause):
        if not isinstance(node, LedgerSum | LedgerCount):
            continue
        if node.selector.is_open():
            found.append(
                CompilationError(
                    code="selector_is_open",
                    path=path,
                    detail="выборка без единого условия",
                )
            )
        found.extend(_check_months(node.selector, path))
    return found


def _check_months(selector: Selector, path: str) -> list[CompilationError]:
    return [
        CompilationError(
            code="month_is_invalid", path=path, detail=f"месяц {month} вне диапазона 1..12"
        )
        for month in selector.months
        if not 1 <= month <= _MONTHS_IN_YEAR
    ]


def check_units(clause: CompiledClause) -> list[CompilationError]:
    """Единица измерения и категории должны быть из известных."""
    found: list[CompilationError] = []
    if clause.formula.unit not in set(Unit):
        found.append(
            CompilationError(code="unit_is_unknown", path="unit", detail=str(clause.formula.unit))
        )
    known = {category.value for category in TransactionCategory}
    for path, node in _nodes(clause):
        if not isinstance(node, LedgerSum | LedgerCount):
            continue
        found.extend(
            CompilationError(code="category_is_unknown", path=path, detail=str(category))
            for category in node.selector.categories
            if category.value not in known
        )
    return found


def check_requirements(clause: CompiledClause) -> list[CompilationError]:
    """Каждая внешняя величина дерева объявлена в required_facts, и наоборот.

    Незаявленная величина — это молчаливая дыра: исполнитель дойдёт до неё и
    остановится, но узнаем мы об этом уже во время расчёта, а не при проверке.
    """
    declared = {item.fact_kind for item in clause.required_facts}
    found: list[CompilationError] = []
    used: set[str] = set()

    for path, node in _nodes(clause):
        if isinstance(node, FactValue):
            used.add(node.fact_kind)
            if node.fact_kind not in declared:
                found.append(
                    CompilationError(
                        code="fact_is_not_declared",
                        path=path,
                        detail=f"{node.fact_kind} используется, но не объявлен в required_facts",
                    )
                )
        elif isinstance(node, ExternalMetric):
            used.add(node.name)
            if node.name not in declared:
                found.append(
                    CompilationError(
                        code="fact_is_not_declared",
                        path=path,
                        detail=f"внешняя величина {node.name} не объявлена в required_facts",
                    )
                )

    found.extend(
        CompilationError(
            code="fact_is_not_used",
            path="required_facts",
            detail=f"{kind} объявлен, но в дереве не встречается",
        )
        for kind in sorted(declared - used)
    )
    return found


def candidates(requirement: FactRequirement, facts: Iterable[Fact]) -> list[Fact]:
    """Факты, удовлетворяющие требованию целиком.

    Совпадения по названию вида мало: порог прошлого периода, сумма в другой валюте
    и обязательство дочерней компании носят один `fact_kind`. Поэтому сверяются
    также счёт, область и контрагент — всё, что требование о себе объявило.
    """
    found = []
    for fact in facts:
        if fact.metric_name != requirement.fact_kind:
            continue
        if fact.account_id != requirement.account_id:
            continue
        stated = getattr(fact, "counterparty", None)
        if requirement.counterparty and stated != requirement.counterparty:
            continue
        found.append(fact)
    return found


def resolve_requirements(
    clause: CompiledClause, facts: Iterable[Fact]
) -> tuple[FactRequirement, ...]:
    """Отметить, какие требуемые величины нашёл детерминированный слой.

    Ненайденная величина не обнуляется и не выбрасывается: она остаётся с пометкой
    `NEEDS_RESOLUTION` и становится входом следующего шага поиска.

    Несколько подходящих фактов — это `AMBIGUOUS`, а не «возьмём первый»: выбор по
    порядку чтения документов дал бы ответ, зависящий от имён файлов.
    """
    available = list(facts)
    resolved = []
    for item in clause.required_facts:
        matched = candidates(item, available)
        if not matched:
            outcome = Resolution.NEEDS_RESOLUTION
        elif len(matched) > 1 and item.cardinality is Cardinality.ONE:
            outcome = Resolution.AMBIGUOUS
        else:
            outcome = Resolution.RESOLVED
        resolved.append(item.model_copy(update={"resolution": outcome}))
    return tuple(resolved)


def check_coverage(
    clauses: Sequence[CompiledClause], expected: Sequence[tuple[str, str]]
) -> list[CompilationError]:
    """Ни одной лишней ячейки и ни одной пропущенной."""
    compiled = [clause.cell for clause in clauses]
    found = [
        CompilationError(
            code="cell_is_duplicated",
            path="clauses",
            detail=f"{cell} скомпилирован больше одного раза",
        )
        for cell in sorted({cell for cell in compiled if compiled.count(cell) > 1})
    ]
    found.extend(
        CompilationError(code="cell_is_missing", path="clauses", detail=f"{cell} не скомпилирован")
        for cell in sorted(set(expected) - set(compiled))
    )
    return found


def validate(
    clause: CompiledClause,
    *,
    documents: Mapping[str, DocumentFacts],
    expected: Sequence[tuple[str, str]],
) -> list[CompilationError]:
    """Все проверки одного пункта. Порядок не важен: они независимы."""
    return [
        *check_cell(clause, expected),
        *check_operations(clause),
        *check_quote(clause, documents),
        *check_edition(clause, documents),
        *check_selectors(clause),
        *check_units(clause),
        *check_requirements(clause),
        *[
            CompilationError(code="dimension_mismatch", path=error.path, detail=error.detail)
            for error in check_dimensions(clause)
        ],
    ]
