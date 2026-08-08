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
from collections.abc import Iterable, Sequence
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
from halyk.llm.runner import Request, Role, StructuredModelRunner
from halyk.llm.schema import strict_schema
from halyk.models.classification import TransactionCategory
from halyk.models.document import DocumentFacts, DocumentKind
from halyk.models.fact import Fact
from halyk.models.formula import (
    CovenantFormula,
    Difference,
    Direction,
    LedgerSum,
    Selector,
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


def _expand_ebitda(value: Any) -> Any:
    """Заменить узел EBITDA стандартным определением из статей реестра."""
    if isinstance(value, dict):
        if value.get("op") == "fact_value" and str(value.get("fact_kind", "")).casefold() in {
            "ebitda",
            "borrower_ebitda",
        }:
            return Difference(
                left=LedgerSum(
                    selector=Selector(
                        categories=(TransactionCategory.REVENUE,), direction=Direction.INFLOW
                    )
                ),
                right=LedgerSum(
                    selector=Selector(
                        categories=(TransactionCategory.OPEX,), direction=Direction.OUTFLOW
                    )
                ),
            ).model_dump(mode="python")
        return {key: _expand_ebitda(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_expand_ebitda(item) for item in value)
    return value


def normalise_derived_metrics(
    clauses: Sequence[CompiledClause], documents: Iterable[DocumentFacts]
) -> tuple[CompiledClause, ...]:
    """Развернуть стандартную EBITDA, если готовое значение нигде не раскрыто."""
    if _ebitda_is_disclosed(documents):
        return tuple(clauses)
    found = []
    for clause in clauses:
        kinds = {item.fact_kind.casefold() for item in clause.required_facts}
        if not kinds.intersection({"ebitda", "borrower_ebitda"}):
            found.append(clause)
            continue
        formula = CovenantFormula.model_validate(
            _expand_ebitda(clause.formula.model_dump(mode="python"))
        )
        requirements = tuple(
            item
            for item in clause.required_facts
            if item.fact_kind.casefold() not in {"ebitda", "borrower_ebitda"}
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

    runner: StructuredModelRunner
    errata: ErrataRegistry = field(default_factory=ErrataRegistry.load)

    def compile(self, batch: CompilerBatch, facts: Sequence[Fact] = ()) -> CompilationResult:
        """Скомпилировать пункты заёмщика и отметить, каких величин им не хватает."""
        documents = index(batch.documents)
        clauses = self.runner.run(
            self.request(batch),
            partial(self._parse, batch=batch, documents=documents),
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
