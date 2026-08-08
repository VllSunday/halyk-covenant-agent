"""Путь одного заёмщика: от его документов до заполненных ячеек.

Порядок стадий вынужденный, а не выбранный.

Компиляция идёт первой, потому что период измерения объявлен в договоре, а от границ
периода зависит состав строк реестра. Взять период из имени файла реестра или из года
в шапке значило бы посчитать по своему календарю, а не по договорному.

Нормализация выполняется дважды. Первый раз — по тому, что прочитал детерминированный
слой; второй — после resolver, который мог принести и величину, и предписание изменить
операцию. Совмещать их в один проход нельзя: до вызова resolver неизвестно, о чём
спрашивать, а после него реестр уже другой, и разрешение требований обязано считаться
заново по обновлённому составу фактов.

Классификация идёт последней из модельных стадий: она смотрит на операции в том виде,
в каком их оставили корректировки, — с подставленной суммой и перенесённой датой.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from halyk.compiler.batch import CompilerBatch
from halyk.compiler.contract import CompiledClause, FactRequirement, Resolution
from halyk.compiler.validator import resolve_requirements
from halyk.execution.executor import Executor, Outcome
from halyk.ingest.inventory import DatasetInventory
from halyk.ingest.normalise import CovenantPeriod, NormalisedLedger, normalise
from halyk.knowledge.classifier import ClassificationResult
from halyk.knowledge.facts import FactSet
from halyk.knowledge.related_party import RelatedPartyError, RelatedPartyIndex, build_index
from halyk.llm.documents import index
from halyk.models.adjustment import Adjustment
from halyk.models.document import DocumentFacts
from halyk.models.fact import Fact, RelatedPartyPolicyFact
from halyk.models.lineage import InvariantCheck
from halyk.models.result import Verdict
from halyk.models.transaction import LedgerRow
from halyk.pipeline.engines import Engines
from halyk.pipeline.report import Note, Problem
from halyk.resolution.batch import ResolutionResult, ResolverBatch
from halyk.resolution.facts import to_facts
from halyk.verification.invariants import check_answer


@dataclass(frozen=True, slots=True)
class AnsweredCell:
    """Посчитанная ячейка вместе с тем, чем она обоснована.

    Вердикт и число лежат отдельно от `Outcome` не для удобства: в исходе они
    необязательны, потому что расчёт мог и не состояться, а сюда попадает уже
    проверенная ячейка, и второй раз спрашивать «а вдруг там пусто» незачем.
    """

    clause: CompiledClause
    outcome: Outcome
    verdict: Verdict
    actual: Decimal
    threshold: Decimal
    invariants: tuple[InvariantCheck, ...]

    @property
    def cell(self) -> tuple[str, str]:
        return self.clause.cell


@dataclass(frozen=True, slots=True)
class BorrowerResult:
    """Итог по заёмщику: что посчитано, что помешало и во что это обошлось."""

    scenario_id: str
    account_id: str
    period: CovenantPeriod | None = None
    answered: tuple[AnsweredCell, ...] = ()
    problems: tuple[Problem, ...] = ()
    notes: tuple[Note, ...] = ()
    transactions: int = 0
    compiler_batches: int = 0
    resolver_batches: int = 0


def with_categories(ledger: NormalisedLedger, classified: ClassificationResult) -> NormalisedLedger:
    """Проставить операциям статью, к которой их отнесли.

    Нерешённая операция сохраняет то, что было: формулировка документа, которую мы не
    смогли отобразить в онтологию, ценнее подставленного `unknown` — по ней видно, что
    именно осталось непонятым.
    """
    decided = {
        record.txn_id: record.final_category
        for record in classified.records
        if record.final_category.is_decided
    }
    transactions = tuple(
        transaction.model_copy(update={"covenant_category": category.value})
        if (category := decided.get(transaction.txn_id)) is not None
        else transaction
        for transaction in ledger.transactions
    )
    return replace(ledger, transactions=transactions)


def _related_index(
    facts: Sequence[Fact], account_id: str
) -> tuple[RelatedPartyIndex | None, list[Note]]:
    """Индекс связанных сторон, если досье разобрано.

    Отсутствие индекса здесь не ошибка: ковенант по связанным сторонам есть не у
    каждого заёмщика, а тот, у кого он есть, остановится на нём сам — исполнитель
    откажется считать фильтр без досье.
    """
    policy = next((f for f in facts if isinstance(f, RelatedPartyPolicyFact)), None)
    if policy is None:
        return None, []
    try:
        return build_index(account_id, policy.as_policy()), []
    except RelatedPartyError as exc:
        return None, [Note(code="related_party_index_failed", subject=account_id, detail=str(exc))]


def _requirement_problem(
    requirement: FactRequirement,
    cell: str,
    ambiguous: frozenset[str],
    refusals: dict[str, str],
) -> Problem:
    """Открытое требование в виде, по которому понятно, что с ним делать.

    Спор и отказ разведены намеренно: первый чинится правилом разрешения, второй —
    поиском документа, которого в наборе, возможно, и нет.
    """
    disputed = (
        requirement.resolution is Resolution.AMBIGUOUS or requirement.requirement_id in ambiguous
    )
    detail = refusals.get(requirement.requirement_id, requirement.description)
    return Problem(
        code="requirement_is_ambiguous" if disputed else "requirement_is_open",
        subject=f"{cell}/{requirement.requirement_id}",
        detail=f"{requirement.fact_kind}: {detail}",
    )


def _ledger_notes(ledger: NormalisedLedger) -> list[Note]:
    return [
        Note(
            code="adjustment_not_applied",
            subject=adjustment.id,
            detail=f"{adjustment.status.value}: {adjustment.selector.describe()}",
        )
        for adjustment in ledger.problems
    ]


def _classification_notes(classified: ClassificationResult) -> list[Note]:
    return [
        Note(code="transaction_without_category", subject=record.txn_id, detail=record.note)
        for record in classified.unresolved
    ]


def _cell_problems(
    clause: CompiledClause, ambiguous: frozenset[str], refusals: dict[str, str]
) -> list[Problem]:
    """Причины, по которым пункт нельзя даже начинать считать."""
    cell = f"{clause.formula.scenario_id}/{clause.formula.clause_id}"
    found = [
        Problem(
            code="clause_is_incomplete",
            subject=cell,
            detail=f"компилятор не перевёл в дерево: {term}",
        )
        for term in clause.unresolved_terms
    ]
    found += [
        _requirement_problem(item, cell, ambiguous, refusals) for item in clause.open_requirements
    ]
    return found


def _answer(
    clause: CompiledClause, executor: Executor
) -> tuple[AnsweredCell | None, Problem | None]:
    """Ячейка или причина, по которой её не будет."""
    cell = f"{clause.formula.scenario_id}/{clause.formula.clause_id}"
    outcome = executor.run(clause.formula)
    if not outcome.is_resolved:
        return None, Problem(
            code="calculation_failed",
            subject=cell,
            detail=f"{outcome.failure}: {outcome.failure_detail} (узел {outcome.failure_path})",
        )
    if outcome.actual is None or outcome.status is None:
        return None, Problem(
            code="cell_is_incomplete", subject=cell, detail="расчёт не дал ни числа, ни вердикта"
        )

    threshold = executor.evaluate(clause.formula.threshold, "threshold").amount
    invariants = check_answer(outcome, clause.formula.operator, threshold)
    if failed := [check for check in invariants if not check.passed]:
        return None, Problem(
            code="invariant_failed",
            subject=cell,
            detail="; ".join(f"{check.name}: {check.detail}" for check in failed),
        )
    return AnsweredCell(
        clause=clause,
        outcome=outcome,
        verdict=Verdict(outcome.status),
        actual=outcome.actual,
        threshold=threshold,
        invariants=invariants,
    ), None


def _resolve_facts(
    account_id: str,
    documents: Sequence[DocumentFacts],
    requirements: Sequence[FactRequirement],
    engines: Engines,
) -> tuple[ResolutionResult, tuple[Fact, ...], list[Problem], int]:
    """Дочитывание открытых требований и перевод ответа в факты расчёта."""
    batch = ResolverBatch.build(account_id, requirements, documents)
    if batch.is_empty:
        return ResolutionResult(account_id=account_id), (), [], 0

    resolution = engines.resolver.resolve(batch)
    facts, errors = to_facts(resolution, batch.requirements, index(documents))
    problems = [
        Problem(code=error.code, subject=error.requirement_id, detail=error.detail)
        for error in errors
    ]
    return resolution, facts, problems, 1


def solve_borrower(
    scenario_id: str,
    account_id: str,
    clause_ids: Iterable[str],
    *,
    inventory: DatasetInventory,
    facts: FactSet,
    engines: Engines,
) -> BorrowerResult:
    """Полный путь заёмщика: компиляция, реестр, дочитывание, статьи, расчёт."""
    documents = inventory.by_account(account_id)
    own_facts = facts.by_account(account_id)
    rows: tuple[LedgerRow, ...] = tuple(
        row for row in inventory.ledger if row.account_id == account_id
    )
    adjustments: tuple[Adjustment, ...] = tuple(
        item for item in facts.adjustments if item.account_id == account_id
    )

    batch = CompilerBatch.build(account_id, scenario_id, clause_ids, documents)
    compiled = engines.compiler.compile(batch, own_facts)
    period = CovenantPeriod(*compiled.period)

    # Первая нормализация: реестр в том виде, в каком его оставил детерминированный
    # слой. Она же показывает, какие предписания документов до операций не дошли.
    ledger = normalise(rows, adjustments, period)

    resolution, resolved_facts, bridge_problems, resolver_batches = _resolve_facts(
        account_id, documents, compiled.open_requirements, engines
    )

    # Повторный проход по обновлённому составу: и реестр, и требования обязаны учесть
    # то, что принёс resolver, иначе его ответ остался бы в отчёте и не в расчёте.
    all_facts = (*own_facts, *resolved_facts)
    ledger = normalise(rows, (*adjustments, *resolution.adjustments), period)
    clauses = tuple(
        clause.model_copy(update={"required_facts": resolve_requirements(clause, all_facts)})
        for clause in compiled.clauses
    )

    classified = engines.classifier.run(ledger, [account_id])
    ledger = with_categories(ledger, classified)

    related, notes = _related_index(all_facts, account_id)
    executor = Executor(
        account_id=account_id,
        transactions=ledger.transactions,
        facts=all_facts,
        related=related,
    )

    ambiguous = frozenset(resolution.ambiguous)
    refusals = {
        item.requirement_id: f"{item.reason.value}: {item.explanation}"
        for item in resolution.unresolved
    }

    answered: list[AnsweredCell] = []
    problems: list[Problem] = list(bridge_problems)
    for clause in clauses:
        if blocking := _cell_problems(clause, ambiguous, refusals):
            problems.extend(blocking)
            continue
        cell, problem = _answer(clause, executor)
        if cell is not None:
            answered.append(cell)
        if problem is not None:
            problems.append(problem)

    notes += _ledger_notes(ledger) + _classification_notes(classified)
    return BorrowerResult(
        scenario_id=scenario_id,
        account_id=account_id,
        period=period,
        answered=tuple(answered),
        problems=tuple(problems),
        notes=tuple(notes),
        transactions=len(ledger.transactions),
        compiler_batches=1,
        resolver_batches=resolver_batches,
    )
