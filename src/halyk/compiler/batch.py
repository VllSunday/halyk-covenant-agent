"""Компиляция всех пунктов одного заёмщика одним запросом.

Батч на заёмщика, а не на пункт: двенадцать запросов укладываются в боевое окно,
четыре сотни — нет. Заодно это единственная форма, в которой модель видит договор
целиком и может заметить, что пункт 6.3 переопределён допсоглашением.

Батч принимается или отклоняется целиком. Частичный приём пришлось бы дополнять
вторым запросом — со своим ключом кэша и своим повтором, — а повтор у нас один и
живёт в Runner. Терять при этом почти нечего: пункты одного заёмщика читаются из
одного договора, и ошибка в одном из них обычно означает, что модель взяла не ту
редакцию, то есть остальные тоже неверны.

Адрес источника мы не принимаем от модели — только имя файла и номер страницы. Всё
остальное (хеш, тип, статус, право менять расчёт) подставляется из переписи: приняв
статус от модели, мы позволили бы ей объявить вытесненную редакцию действующей и
обойти проверку, ради которой эта проверка написана.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from typing import Any

from halyk.compiler.contract import (
    INSTRUCTIONS,
    OUTPUT_CONTRACT,
    CompiledClause,
    CompilerResponse,
    FactRequirement,
)
from halyk.compiler.validator import (
    CompilationError,
    check_coverage,
    resolve_requirements,
    validate,
)
from halyk.knowledge.authority import resolve_authority
from halyk.knowledge.errata import ErrataRegistry
from halyk.llm.documents import index, own_documents, render, source_hashes
from halyk.llm.runner import ModelRunner, Request, Role
from halyk.llm.schema import strict_schema
from halyk.models.classification import TransactionCategory
from halyk.models.document import DocumentFacts, DocumentKind
from halyk.models.fact import Fact
from halyk.models.formula import (
    CovenantFormula,
    Difference,
    Direction,
    FactValue,
    LedgerSum,
    Ratio,
    Selector,
    Sum,
)
from halyk.models.source import SourceAuthority

SCHEMA_NAME = "compiler_response"

# Статьи, которыми компилятор имеет право пользоваться в селекторах. Список закрыт и
# приходит из того же перечисления, что и разбор реестра: модель, придумавшая статью,
# не пройдёт даже схему.
CATEGORIES = tuple(
    category.value
    for category in TransactionCategory
    if category is not TransactionCategory.UNKNOWN
)

# Адрес, который заведомо не совпадёт ни с одним документом переписи. Нужен, чтобы
# ответ со ссылкой на несуществующий файл дошёл до валидатора и получил внятный код
# отказа, а не упал на длине хеша внутри pydantic.
_UNKNOWN_HASH = "0" * 64

_DISCLOSED_EBITDA = re.compile(
    r"\bEBITDA\b.{0,100}(?:amounts?\s+to|состав(?:ляет|ила)|равн\w*)\s*"
    r"(?:USD\s*|\$\s*)?[0-9]",
    re.IGNORECASE | re.DOTALL,
)

_EBITDA_WORD = re.compile(r"\bEBITDA\b", re.IGNORECASE)
_EBITDA_DETAIL_CATEGORIES = {
    TransactionCategory.OPEX.value,
    TransactionCategory.PAYROLL.value,
    TransactionCategory.UTILITIES.value,
    TransactionCategory.RENT.value,
    TransactionCategory.INSURANCE_PREMIUM.value,
}
_MIN_INLINE_EBITDA_TERMS = 2


class ClauseRejectedError(ValueError):
    """Ответ компилятора не прошёл проверку. Причины — списком, с адресами узлов."""

    def __init__(self, errors: Sequence[CompilationError]) -> None:
        super().__init__("; ".join(f"{item.code}@{item.path}" for item in errors))
        self.errors = tuple(errors)


@dataclass(frozen=True, slots=True)
class CompilerBatch:
    """Работа одного заёмщика: какие ячейки закрыть и из каких документов."""

    account_id: str
    scenario_id: str
    clause_ids: tuple[str, ...]
    documents: tuple[DocumentFacts, ...]

    @classmethod
    def build(
        cls,
        account_id: str,
        scenario_id: str,
        clause_ids: Iterable[str],
        documents: Iterable[DocumentFacts],
    ) -> CompilerBatch:
        return cls(
            account_id=account_id,
            scenario_id=scenario_id,
            clause_ids=tuple(sorted(set(clause_ids))),
            documents=own_documents(account_id, documents),
        )

    @property
    def cells(self) -> tuple[tuple[str, str], ...]:
        return tuple((self.scenario_id, clause_id) for clause_id in self.clause_ids)

    def payload(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "scenario_id": self.scenario_id,
            "clauses": list(self.clause_ids),
            "categories": list(CATEGORIES),
            "documents": render(self.documents),
        }


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Пункты заёмщика в исполнимой форме вместе с тем, чего им не хватает."""

    account_id: str
    clauses: tuple[CompiledClause, ...]

    @property
    def period(self) -> tuple[date, date]:
        return self.clauses[0].period_start, self.clauses[0].period_end

    @property
    def open_requirements(self) -> tuple[FactRequirement, ...]:
        return tuple(item for clause in self.clauses for item in clause.open_requirements)


def _incomplete_clauses(clauses: tuple[CompiledClause, ...]) -> str | None:
    cells = [
        f"{clause.formula.scenario_id}/{clause.formula.clause_id}"
        for clause in clauses
        if clause.unresolved_terms
    ]
    if not cells:
        return None
    return f"неразобранные термины в ячейках: {', '.join(cells)}"


def normalise_periods(clauses: Sequence[CompiledClause]) -> tuple[CompiledClause, ...]:
    """Снять одиночную ошибку периода по большинству пунктов одного договора."""
    if not clauses:
        return ()
    counts = Counter((clause.period_start, clause.period_end) for clause in clauses)
    period, votes = counts.most_common(1)[0]
    if votes <= len(clauses) // 2:
        return tuple(clauses)
    start, end = period
    return tuple(
        clause.model_copy(update={"period_start": start, "period_end": end}) for clause in clauses
    )


def _ebitda_is_disclosed(documents: Iterable[DocumentFacts]) -> bool:
    """Есть ли готовое значение EBITDA вне самого договора."""
    return any(
        document.kind is not DocumentKind.LOAN_AGREEMENT and _DISCLOSED_EBITDA.search(page.text)
        for document in documents
        for page in document.pages
    )


def _ebitda_node() -> Difference:
    return Difference(
        left=LedgerSum(
            selector=Selector(categories=(TransactionCategory.REVENUE,), direction=Direction.INFLOW)
        ),
        right=LedgerSum(
            selector=Selector(categories=(TransactionCategory.OPEX,), direction=Direction.OUTFLOW)
        ),
    )


def _repayments_node() -> LedgerSum:
    return LedgerSum(
        selector=Selector(
            categories=(TransactionCategory.PRINCIPAL_REPAYMENT,), direction=Direction.OUTFLOW
        )
    )


def _debt_node() -> Difference:
    """Долг за период: привлечённое финансирование за вычетом погашений тела.

    Определение не наше: так его записывают сами договоры набора — «aggregate
    principal amount of Financial Indebtedness drawn during the period less all
    scheduled principal repayments made in that period». Часть договоров ссылается
    на пункт с определением, которого в них нет, и без подстановки величина уходит
    во внешние факты, где её никто не раскрывал.
    """
    return Difference(
        left=LedgerSum(
            selector=Selector(
                categories=(TransactionCategory.FINANCING_INFLOW,), direction=Direction.INFLOW
            )
        ),
        right=_repayments_node(),
    )


# Показатели, состав которых договор называет сам. Величина, собираемая из операций
# реестра, внешним фактом быть не может: спрашивать её у документов бессмысленно, там
# её нет и не должно быть.
def _one_off_addback_node() -> FactValue:
    """Разовые статьи, возвращаемые в EBITDA.

    Это те самые статьи, которые примечания и отчёт аудитора уже перечислили: спрашивать
    их сумму отдельной величиной незачем, она собирается из прочитанных фактов. Порог
    существенности применяется здесь же — иначе в EBITDA вернётся всё упомянутое.
    """
    return FactValue(fact_kind="one_off_item", above_one_off_policy=True)


def _group_capex_node() -> FactValue:
    """Капитальные затраты Группы: поступления основных средств из консолидированной
    отчётности. Отдельной проводки у них нет, величина выводится из движения."""
    return FactValue(fact_kind="ppe_roll_forward")


_STANDARD_DEFINITIONS: dict[str, Callable[[], Sum | Difference | Ratio | LedgerSum | FactValue]] = {
    "ebitda": _ebitda_node,
    "borrower_ebitda": _ebitda_node,
    "total_debt": _debt_node,
    "borrower_total_debt": _debt_node,
    "net_debt": _debt_node,
    "total_indebtedness": _debt_node,
    "financial_indebtedness": _debt_node,
    "scheduled_principal_repayments": _repayments_node,
    "principal_repayments": _repayments_node,
    "debt_to_ebitda_ratio": lambda: Ratio(numerator=_debt_node(), denominator=_ebitda_node()),
    "leverage_ratio": lambda: Ratio(numerator=_debt_node(), denominator=_ebitda_node()),
    "net_leverage_ratio": lambda: Ratio(numerator=_debt_node(), denominator=_ebitda_node()),
    "ebitda_one_off_addback": _one_off_addback_node,
    "ebitda_one_off_addbacks": _one_off_addback_node,
    "one_off_ebitda_adjustment": _one_off_addback_node,
    "one_off_ebitda_adjustments": _one_off_addback_node,
    "auditor_agreed_one_off_ebitda_adjustment": _one_off_addback_node,
    "auditor_agreed_one_off_ebitda_adjustments": _one_off_addback_node,
    "auditor_agreed_one_off_adjustments": _one_off_addback_node,
    "consolidated_capital_expenditures": _group_capex_node,
    "group_consolidated_capex": _group_capex_node,
    "group_capital_expenditures": _group_capex_node,
}


def _standard_for(kind: str) -> Callable[[], Any] | None:
    """Стандартное определение величины по её имени, если оно у нас есть.

    Только точное совпадение. Поиск по признакам («что-то про разовые статьи»)
    пробовался и оказался вреднее пропуска: он перехватывал требования, которые
    resolver закрывал сам, и подменял их пустой суммой.
    """
    return _STANDARD_DEFINITIONS.get(kind)


def _expand_standard(value: Any, kinds: Collection[str]) -> Any:
    """Заменить узлы названных величин их стандартным определением."""
    if isinstance(value, dict):
        kind = str(value.get("fact_kind") or value.get("name") or "").casefold()
        build = _standard_for(kind) if kind in kinds else None
        if value.get("op") in ("fact_value", "external") and build is not None:
            return build().model_dump(mode="python")
        return {key: _expand_standard(item, kinds) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_expand_standard(item, kinds) for item in value)
    return value


def _single_category_sum(value: Any, category: str, direction: str) -> bool:
    if not isinstance(value, dict) or value.get("op") != "ledger_sum":
        return False
    selector = value.get("selector")
    return (
        isinstance(selector, dict)
        and selector.get("categories") == [category]
        and selector.get("direction") == direction
    )


def _outflow_category(value: Any) -> str | None:
    if not isinstance(value, dict) or value.get("op") != "ledger_sum":
        return None
    selector = value.get("selector")
    if not isinstance(selector, dict):
        return None
    categories = selector.get("categories")
    if (
        not isinstance(categories, list)
        or len(categories) != 1
        or not isinstance(categories[0], str)
        or selector.get("direction") != Direction.OUTFLOW.value
    ):
        return None
    return categories[0]


def _collapse_inline_ebitda(value: Any) -> Any:
    """Свести ошибочно детализированную EBITDA к строкам REVENUE минус OPEX."""
    if isinstance(value, list | tuple):
        return type(value)(_collapse_inline_ebitda(item) for item in value)
    if not isinstance(value, dict):
        return value

    repaired = {key: _collapse_inline_ebitda(item) for key, item in value.items()}
    right = repaired.get("right")
    terms = right.get("terms") if isinstance(right, dict) and right.get("op") == "add" else None
    is_candidate = (
        repaired.get("op") == "sub"
        and _single_category_sum(
            repaired.get("left"), TransactionCategory.REVENUE.value, Direction.INFLOW.value
        )
        and isinstance(terms, list)
        and len(terms) >= _MIN_INLINE_EBITDA_TERMS
    )
    if not is_candidate or not isinstance(terms, list):
        return repaired

    categories = [_outflow_category(term) for term in terms]
    opex = [
        term
        for term in terms
        if _single_category_sum(term, TransactionCategory.OPEX.value, Direction.OUTFLOW.value)
    ]
    if (
        None not in categories
        and not set(categories) - _EBITDA_DETAIL_CATEGORIES
        and len(opex) == 1
    ):
        repaired["right"] = opex[0]
    return repaired


def normalise_derived_metrics(
    clauses: Sequence[CompiledClause], documents: Iterable[DocumentFacts]
) -> tuple[CompiledClause, ...]:
    """Развернуть показатели, состав которых договор задаёт сам.

    EBITDA разворачивается только когда готового значения нигде нет: раскрытое в
    примечаниях считается более сильным источником, чем наша реконструкция. У
    долговых величин такого источника не бывает — они по определению собираются из
    движений периода, — поэтому они разворачиваются всегда.
    """
    skip = set()
    if _ebitda_is_disclosed(documents):
        skip = {"ebitda", "borrower_ebitda"}
    found = []
    for clause in clauses:
        formula_payload = clause.formula.model_dump(mode="json")
        if "ebitda" not in skip and _EBITDA_WORD.search(
            f"{clause.formula.title} {clause.formula.quote}"
        ):
            formula_payload = _collapse_inline_ebitda(formula_payload)
        expandable = {
            kind
            for item in clause.required_facts
            if (kind := item.fact_kind.casefold()) not in skip and _standard_for(kind) is not None
        }
        if not expandable:
            formula = CovenantFormula.model_validate(formula_payload)
            found.append(clause.model_copy(update={"formula": formula}))
            continue
        formula = CovenantFormula.model_validate(_expand_standard(formula_payload, expandable))
        requirements = tuple(
            item for item in clause.required_facts if item.fact_kind.casefold() not in expandable
        )
        found.append(clause.model_copy(update={"formula": formula, "required_facts": requirements}))
    return tuple(found)


def check_authority(clause: CompiledClause) -> list[CompilationError]:
    """Пункт читается из документа, на который вообще можно опираться.

    Проверка отделена от `check_edition` не из аккуратности: вытеснённая редакция —
    один частный случай, а посторонний документ и черновик — другие два, и попадают
    они сюда одинаково.
    """
    return [
        CompilationError(
            code="source_is_not_authoritative",
            path=f"source_refs[{position}]",
            detail=f"{ref.file_name}: {ref.authority.value}",
        )
        for position, ref in enumerate(clause.formula.source_refs)
        if ref.authority is not SourceAuthority.AUTHORITATIVE
    ]


def check_period(clauses: Sequence[CompiledClause]) -> list[CompilationError]:
    """Период измерения у всех пунктов заёмщика один.

    Он приходит из договора, но общий для всех его пунктов: расхождение означает не
    разные периоды, а неверно прочитанную дату — и дальше поедет всё, потому что
    состав строк реестра зависит от границ периода.
    """
    periods = {(clause.period_start, clause.period_end) for clause in clauses}
    if len(periods) <= 1:
        return []
    named = ", ".join(f"{start}..{end}" for start, end in sorted(periods))
    return [
        CompilationError(
            code="period_is_inconsistent",
            path="clauses",
            detail=f"пункты одного заёмщика измеряются разными периодами: {named}",
        )
    ]


def _source_ref(raw: Any, documents: dict[str, DocumentFacts]) -> dict[str, Any]:
    """Адрес источника, собранный из переписи по названному моделью файлу."""
    stated = raw if isinstance(raw, dict) else {}
    page = stated.get("page")
    name = stated.get("file_name")
    document = documents.get(name) if isinstance(name, str) else None
    if document is None:
        return {
            "file_hash": _UNKNOWN_HASH,
            "file_name": name if isinstance(name, str) else "",
            "page": page,
        }
    return {
        "file_hash": document.sha256,
        "file_name": document.file_name,
        "page": page,
        "kind": document.kind.value,
        "status": document.status.value,
        "authority": resolve_authority(document).value,
        "account_id": document.account_id,
        "report_number": document.report_number,
        "quote": stated.get("quote"),
    }


def with_known_sources(payload: Any, documents: dict[str, DocumentFacts]) -> Any:
    """Заменить адреса источников на наши до разбора ответа.

    Работает по сырому ответу, а не по разобранной модели: `SourceRef` требует хеша
    длиной в шестьдесят четыре знака, и без подстановки ответ не дошёл бы до
    валидатора вовсе — он падал бы на длине строки, не назвав настоящей причины.
    """
    if not isinstance(payload, dict):
        return payload
    clauses = payload.get("clauses")
    if not isinstance(clauses, list):
        return payload

    repaired = []
    for clause in clauses:
        formula = clause.get("formula") if isinstance(clause, dict) else None
        if not isinstance(formula, dict) or not isinstance(
            refs := formula.get("source_refs"), list
        ):
            repaired.append(clause)
            continue
        known = [_source_ref(ref, documents) for ref in refs]
        repaired.append(clause | {"formula": formula | {"source_refs": known}})
    return payload | {"clauses": repaired}


@dataclass(slots=True)
class CovenantCompiler:
    """Компилятор поверх единственной двери к модели.

    Собственных повторов у него нет: смысловой повтор один на запрос и делается
    Runner, для которого невалидный ответ — это исключение из разбора. Поэтому
    проверка ответа и живёт в `_parse`, а не после вызова.
    """

    runner: ModelRunner
    errata: ErrataRegistry = field(default_factory=ErrataRegistry.load)

    def compile(self, batch: CompilerBatch, facts: Sequence[Fact] = ()) -> CompilationResult:
        """Скомпилировать пункты заёмщика и отметить, каких величин им не хватает."""
        documents = index(batch.documents)
        clauses = self.runner.run(
            self.request(batch),
            partial(self._parse, batch=batch, documents=documents),
            escalate_if=_incomplete_clauses,
        )
        clauses = tuple(
            clause.model_copy(update={"formula": self.errata.apply_formula(clause.formula)[0]})
            for clause in clauses
        )
        # Разрешение требований детерминировано и от ответа модели не зависит, поэтому
        # делается после разбора: иначе оно повторялось бы на каждой попытке и меняло
        # бы причину отказа вместе с составом фактов.
        resolved = tuple(
            clause.model_copy(update={"required_facts": resolve_requirements(clause, facts)})
            for clause in clauses
        )
        return CompilationResult(account_id=batch.account_id, clauses=resolved)

    def request(self, batch: CompilerBatch) -> Request:
        return Request(
            role=Role.COMPILER,
            account_id=batch.account_id,
            instructions=INSTRUCTIONS,
            schema=strict_schema(CompilerResponse),
            schema_name=SCHEMA_NAME,
            contract=OUTPUT_CONTRACT,
            payload=batch.payload(),
            source_hashes=source_hashes(batch.documents),
        )

    @staticmethod
    def _parse(
        payload: dict[str, Any], *, batch: CompilerBatch, documents: dict[str, DocumentFacts]
    ) -> tuple[CompiledClause, ...]:
        response = CompilerResponse.model_validate(with_known_sources(payload, documents))
        clauses = normalise_periods(tuple(sorted(response.clauses, key=lambda clause: clause.cell)))
        clauses = normalise_derived_metrics(clauses, documents.values())

        errors = [*check_coverage(clauses, batch.cells), *check_period(clauses)]
        for clause in clauses:
            errors += validate(clause, documents=documents, expected=batch.cells)
            errors += check_authority(clause)
        if errors:
            raise ClauseRejectedError(errors)
        return clauses
