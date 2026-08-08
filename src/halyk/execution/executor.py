"""Детерминированный исполнитель Formula AST.

Считает только код: дерево пришло от модели, но ни одно решение здесь не спрашивает
её снова. Вся арифметика идёт в `Decimal` — промежуточный `float` на суммах в
десятки миллионов теряет копейки, а `actual` сравнивается с эталоном по
относительной погрешности.

Отказ типизирован и никогда не подменяется числом. Величина, которой в наборе нет,
обязана остановить ячейку: подставленная догадка дала бы верный ответ на открытом
наборе и молчаливо неверный на приватном.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from halyk.knowledge.kyc import normalise_counterparty
from halyk.knowledge.related_party import RelatedPartyError, RelatedPartyIndex
from halyk.models.adjustment import NormalisedTransaction
from halyk.models.classification import TransactionCategory
from halyk.models.covenant import Operator
from halyk.models.fact import (
    AggregateObligationFact,
    Fact,
    OneOffItemFact,
    OneOffPolicyFact,
    PpeRollForwardFact,
)
from halyk.models.formula import (
    Absolute,
    Condition,
    Constant,
    CovenantFormula,
    Difference,
    Direction,
    EvidenceMode,
    ExternalMetric,
    FactValue,
    Largest,
    LedgerCount,
    LedgerSum,
    Product,
    Quantity,
    Ratio,
    RelatedParty,
    Selector,
    Smallest,
    Sum,
)
from halyk.money import quantize


class Failure(StrEnum):
    """Причина, по которой ячейка не может быть посчитана."""

    MISSING_EXTERNAL_METRIC = "missing_external_metric"
    MISSING_FACT = "missing_fact"
    MISSING_KYC_POLICY = "missing_kyc_policy"
    AMBIGUOUS_COUNTERPARTY = "ambiguous_counterparty"
    ZERO_DENOMINATOR = "zero_denominator"

    @property
    def means_we_missed_a_document(self) -> bool:
        """Отказ, которого на этих данных быть не должно.

        Организаторы подтвердили, что всё нужное лежит в наборе и внешних источников
        не требуется ни в публичном, ни в приватном. Значит ссылка на невыведенную
        внешнюю величину — признак непрочитанного документа с нашей стороны, а не
        свойства данных, и в боевом прогоне обязана валить строгую проверку.

        Ровно так мы и потеряли капитальные затраты Группы: англоязычный отчёт без
        номера счёта уходил в шум, а ячейка молча оставалась несчитанной.
        """
        return self is Failure.MISSING_EXTERNAL_METRIC


class Diagnostic(StrEnum):
    """То, что стоит показать в отчёте, но что само по себе не ошибка."""

    EMPTY_SELECTION = "empty_selection"


class ExecutionError(RuntimeError):
    """Отказ с адресом узла: по нему видно, какая часть формулы не сошлась."""

    def __init__(self, reason: Failure, path: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail} (узел {path})")
        self.reason = reason
        self.path = path
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Value:
    """Число вместе с тем, из чего оно получилось."""

    amount: Decimal
    rows: frozenset[str] = frozenset()
    facts: frozenset[str] = frozenset()
    diagnostics: frozenset[tuple[Diagnostic, str]] = frozenset()


@dataclass(frozen=True, slots=True)
class Verdict:
    """Число и вердикт без улики. Промежуточная форма для контрфактического перебора."""

    actual: Decimal
    breached: bool
    triggered: bool
    rows: frozenset[str]
    facts: frozenset[str]
    trigger_rows: frozenset[str]
    trigger_facts: frozenset[str]
    diagnostics: frozenset[tuple[Diagnostic, str]]


@dataclass(frozen=True, slots=True)
class Outcome:
    """Ответ по одной ячейке."""

    scenario_id: str
    clause_id: str
    actual: Decimal | None
    status: str | None
    evidence_txn_id: str | None
    rows: tuple[str, ...]
    facts: tuple[str, ...]
    diagnostics: tuple[tuple[Diagnostic, str], ...]
    # Строки и факты условия срабатывания держатся отдельно от строк расчёта: в
    # `actual` они не входят, но без них не объяснить, почему пункт применён.
    trigger_rows: tuple[str, ...] = ()
    trigger_facts: tuple[str, ...] = ()
    triggered: bool = True
    failure: Failure | None = None
    failure_path: str | None = None
    failure_detail: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.failure is None


@dataclass(slots=True)
class Executor:
    """Расчёт по одному заёмщику.

    Индекс связанных сторон приходит готовым: сопоставление названий — отдельный
    этап, и решать его посреди арифметики исполнитель не должен.
    """

    account_id: str
    transactions: Sequence[NormalisedTransaction]
    facts: Sequence[Fact]
    related: RelatedPartyIndex | None = None
    excluded: frozenset[str] = field(default_factory=frozenset)

    def with_excluded(self, txn_ids: frozenset[str]) -> Executor:
        """Копия исполнителя без указанных операций — основа контрфактической улики."""
        return Executor(
            account_id=self.account_id,
            transactions=self.transactions,
            facts=self.facts,
            related=self.related,
            excluded=self.excluded | txn_ids,
        )

    # --- отбор строк ---------------------------------------------------------

    def _one_off_minimum(self) -> Decimal | None:
        for fact in self.facts:
            if isinstance(fact, OneOffPolicyFact):
                return fact.minimum.to_decimal()
        return None

    def _selected(self, selector: Selector, path: str) -> list[NormalisedTransaction]:
        """Строки, прошедшие селектор, в порядке их идентификаторов.

        Сортировка не косметика: без неё сумма зависела бы от порядка чтения файла,
        а `Decimal` при сложении в разном порядке даёт разные хвосты.
        """
        chosen: list[NormalisedTransaction] = []
        for transaction in self.transactions:
            if not self._passes(transaction, selector):
                continue
            # Связанность проверяется последней: у строки, отсеянной раньше,
            # неразрешимый контрагент не должен ронять прогон.
            if selector.related_party is not RelatedParty.ANY and not self._related_ok(
                transaction, selector.related_party, path
            ):
                continue
            chosen.append(transaction)
        return sorted(chosen, key=lambda t: t.txn_id)

    def _passes(self, transaction: NormalisedTransaction, selector: Selector) -> bool:
        """Все условия селектора, кроме связанности.

        Строка без суммы не отбрасывается как нулевая: пустая сумма означает, что
        фактическое значение раскрыто в другом документе, и если оно не подставлено
        корректировкой, то в расчёт эта строка войти не может.
        """
        if transaction.amount is None:
            return False

        amount = transaction.amount.to_decimal()
        return all(
            (
                not transaction.excluded,
                transaction.in_period,
                transaction.txn_id not in self.excluded,
                transaction.txn_id not in selector.exclude_txn_ids,
                transaction.row.account_id == self.account_id,
                not selector.categories
                or transaction.covenant_category in {c.value for c in selector.categories},
                not selector.months or transaction.effective_date.month in selector.months,
                # Названия сверяются в нормализованной форме, той же, что у связанных
                # сторон: в реестре и в договоре одна компания пишется по-разному
                # («Aktau Holdings LLP» против «Aktau Holdings L.L.P.»). Сходство здесь
                # не используется — только точное совпадение после нормализации.
                not selector.counterparties
                or normalise_counterparty(transaction.row.counterparty)
                in {normalise_counterparty(name) for name in selector.counterparties},
                selector.direction is not Direction.OUTFLOW or amount < 0,
                selector.direction is not Direction.INFLOW or amount > 0,
            )
        )

    def _related_ok(
        self, transaction: NormalisedTransaction, mode: RelatedParty, path: str
    ) -> bool:
        if self.related is None:
            raise ExecutionError(
                Failure.MISSING_KYC_POLICY,
                path,
                f"{self.account_id}: фильтр по связанным сторонам без разобранного досье",
            )
        try:
            match = self.related.require(self.related.classify(transaction.row.counterparty))
        except RelatedPartyError as exc:
            raise ExecutionError(Failure.AMBIGUOUS_COUNTERPARTY, path, str(exc)) from exc
        return match.is_related if mode is RelatedParty.ONLY else not match.is_related

    # --- вычисление ----------------------------------------------------------

    def evaluate(self, node: Quantity, path: str = "measure") -> Value:
        """Значение узла.

        Листья и составные узлы разведены: первые читают данные, вторые только
        складывают уже посчитанное. Отказ возможен и там, и там, но по разным
        причинам — лист не находит источника, составной узел делит на ноль.
        """
        if isinstance(node, LedgerSum | LedgerCount | Constant | ExternalMetric | FactValue):
            return self._leaf(node, path)
        return self._composite(node, path)

    def _leaf(
        self,
        node: LedgerSum | LedgerCount | Constant | ExternalMetric | FactValue,
        path: str,
    ) -> Value:
        """Узел, читающий данные: реестр, факты или объявленную константу."""
        match node:
            case LedgerSum():
                rows = self._selected(node.selector, path)
                # Модуль: расходы в реестре отрицательны, а `actual` по условию задачи
                # всегда положителен. Направление уже задано селектором.
                total = sum(
                    (abs(r.amount.to_decimal()) for r in rows if r.amount is not None), Decimal(0)
                )
                return Value(
                    amount=total,
                    rows=frozenset(r.txn_id for r in rows),
                    diagnostics=_empty_if(rows, path),
                )
            case LedgerCount():
                rows = self._selected(node.selector, path)
                return Value(
                    amount=Decimal(len(rows)),
                    rows=frozenset(r.txn_id for r in rows),
                    diagnostics=_empty_if(rows, path),
                )
            case Constant():
                return Value(amount=node.value)
            case ExternalMetric():
                raise ExecutionError(
                    Failure.MISSING_EXTERNAL_METRIC,
                    path,
                    f"{node.name}: {node.description}. Величина в наборе не раскрыта, "
                    f"подставлять её нельзя",
                )
            case FactValue():
                return self._fact_value(node, path)

    def _composite(
        self,
        node: Sum | Difference | Product | Ratio | Absolute | Largest | Smallest,
        path: str,
    ) -> Value:
        """Узел, складывающий уже посчитанное. Данных сам не читает."""
        match node:
            case Sum():
                parts = [self.evaluate(t, f"{path}.terms[{i}]") for i, t in enumerate(node.terms)]
                return _combine(sum((p.amount for p in parts), Decimal(0)), parts)
            case Difference():
                left = self.evaluate(node.left, f"{path}.left")
                right = self.evaluate(node.right, f"{path}.right")
                return _combine(left.amount - right.amount, [left, right])
            case Product():
                parts = [
                    self.evaluate(f, f"{path}.factors[{i}]") for i, f in enumerate(node.factors)
                ]
                product = Decimal(1)
                for part in parts:
                    product *= part.amount
                return _combine(product, parts)
            case Ratio():
                numerator = self.evaluate(node.numerator, f"{path}.numerator")
                denominator = self.evaluate(node.denominator, f"{path}.denominator")
                if denominator.amount == 0:
                    raise ExecutionError(
                        Failure.ZERO_DENOMINATOR,
                        f"{path}.denominator",
                        "знаменатель равен нулю, отношение не определено",
                    )
                return _combine(numerator.amount / denominator.amount, [numerator, denominator])
            case Absolute():
                inner = self.evaluate(node.value, f"{path}.value")
                return _combine(abs(inner.amount), [inner])
            case Largest() | Smallest():
                parts = [self.evaluate(v, f"{path}.values[{i}]") for i, v in enumerate(node.values)]
                pick = max if isinstance(node, Largest) else min
                return _combine(pick(p.amount for p in parts), parts)

    def _fact_value(self, node: FactValue, path: str) -> Value:
        # Отсутствие порога и порог, равный нулю, — разные вещи. Без него в EBITDA
        # попадёт всё упомянутое в примечаниях, и число выйдет завышенным, но
        # правдоподобным. Такое молчание дороже остановки.
        minimum = self._one_off_minimum() if node.above_one_off_policy else None
        if node.above_one_off_policy and minimum is None:
            raise ExecutionError(
                Failure.MISSING_FACT,
                path,
                f"{self.account_id}: формула ссылается на порог существенности "
                f"разовых статей, но он в документах не раскрыт",
            )
        total = Decimal(0)
        used: set[str] = set()
        for fact in self.facts:
            if fact.kind != node.fact_kind:
                continue
            # Движение основных средств входит в формулу выведенной величиной:
            # отдельной строки капитальных затрат в отчётности нет, она получается
            # из тождества. Сами движения остаются в факте, чтобы вывод был проверяем.
            if isinstance(fact, PpeRollForwardFact):
                total += fact.additions
                used.add(f"{fact.kind}:additions")
                continue
            if not isinstance(fact, OneOffItemFact | AggregateObligationFact):
                continue
            if node.description_contains and node.description_contains not in fact.description:
                continue
            counterparty = getattr(fact, "counterparty", None)
            if node.counterparty and counterparty != node.counterparty:
                continue
            amount = abs(fact.amount.to_decimal())
            # Порог существенности объявлен в примечаниях и отсекает мелкие статьи:
            # без него EBITDA раздувается на всё, что упомянуто в тексте.
            if minimum is not None and amount < minimum:
                continue
            total += amount
            used.add(f"{fact.kind}:{fact.description}")

        if not used and not node.above_one_off_policy:
            raise ExecutionError(
                Failure.MISSING_FACT,
                path,
                f"факт {node.fact_kind} у {self.account_id} не найден",
            )
        return Value(amount=total, facts=frozenset(used))

    def check(self, condition: Condition, path: str) -> tuple[bool, Value, Value]:
        left = self.evaluate(condition.left, f"{path}.left")
        right = self.evaluate(condition.right, f"{path}.right")
        return condition.operator.holds(left.amount, right.amount), left, right

    # --- ячейка целиком ------------------------------------------------------

    def run(self, formula: CovenantFormula) -> Outcome:
        try:
            return self._run(formula)
        except ExecutionError as exc:
            return Outcome(
                scenario_id=formula.scenario_id,
                clause_id=formula.clause_id,
                actual=None,
                status=None,
                evidence_txn_id=None,
                rows=(),
                facts=(),
                diagnostics=(),
                failure=exc.reason,
                failure_path=exc.path,
                failure_detail=exc.detail,
            )

    def _verdict(self, formula: CovenantFormula) -> Verdict:
        """Число и вердикт без выбора улики.

        Отделено от `_run` намеренно: контрфактический перебор пересчитывает ячейку
        по разу на строку, и если бы каждый пересчёт снова искал улику, работа росла
        бы факториально от числа строк расчёта.
        """
        measured = self.evaluate(formula.measure)
        threshold = self.evaluate(formula.threshold, "threshold")

        triggered = True
        trigger_rows: frozenset[str] = frozenset()
        trigger_facts: frozenset[str] = frozenset()
        trigger_notes: frozenset[tuple[Diagnostic, str]] = frozenset()
        if formula.applies_when is not None:
            triggered, left, right = self.check(formula.applies_when, "applies_when")
            # Строки условия входят в объяснение наравне со строками расчёта: без них
            # непонятно, почему пункт сработал или не сработал.
            trigger_rows = left.rows | right.rows
            trigger_facts = left.facts | right.facts
            trigger_notes = left.diagnostics | right.diagnostics

        # Округление до сравнения, а не после: эталон сверяется с числом, которое мы
        # заявили, и вердикт обязан относиться к нему же.
        actual = quantize(measured.amount, formula.rounding.scale, formula.rounding.mode)
        holds = formula.operator.holds(actual, threshold.amount)
        return Verdict(
            actual=actual,
            breached=triggered and not holds,
            triggered=triggered,
            rows=measured.rows,
            facts=measured.facts,
            trigger_rows=trigger_rows,
            trigger_facts=trigger_facts,
            diagnostics=measured.diagnostics | threshold.diagnostics | trigger_notes,
        )

    def _run(self, formula: CovenantFormula) -> Outcome:
        verdict = self._verdict(formula)

        evidence = None
        if verdict.breached and formula.evidence is EvidenceMode.COUNTERFACTUAL:
            evidence = self._evidence(formula, verdict.rows)

        return Outcome(
            scenario_id=formula.scenario_id,
            clause_id=formula.clause_id,
            actual=verdict.actual,
            status="BREACH" if verdict.breached else "COMPLIANT",
            evidence_txn_id=evidence,
            rows=tuple(sorted(verdict.rows)),
            facts=tuple(sorted(verdict.facts)),
            trigger_rows=tuple(sorted(verdict.trigger_rows)),
            trigger_facts=tuple(sorted(verdict.trigger_facts)),
            diagnostics=tuple(sorted(verdict.diagnostics)),
            triggered=verdict.triggered,
        )

    def _evidence(self, formula: CovenantFormula, rows: frozenset[str]) -> str | None:
        """Операция, без которой нарушения не было бы.

        Перебор контрфактический: строка признаётся уликой, только если её изъятие
        переворачивает вердикт. Правил вида «самая крупная» здесь нет — условие
        задачи прямо отвергает их, и вклад в сумму уликой не делает.

        Если вердикт переворачивают несколько строк, единственной причины нет, и
        ответом остаётся `null`: угадывать здесь дешевле не станет, а неверный
        идентификатор стоит столько же, сколько пустой.
        """
        culprits = []
        for txn_id in sorted(rows):
            without = self.with_excluded(frozenset({txn_id}))
            try:
                # Только вердикт: вложенный поиск улики здесь ничего не добавил бы,
                # а работу умножил бы на число строк ещё раз.
                probe = without._verdict(formula)
            except ExecutionError:
                # Изъятие строки сделало расчёт невозможным — значит она не улика,
                # а несущая конструкция расчёта.
                continue
            if not probe.breached:
                culprits.append(txn_id)
        return culprits[0] if len(culprits) == 1 else None


def _empty_if(rows: Sequence[object], path: str) -> frozenset[tuple[Diagnostic, str]]:
    """Пустая выборка — не ошибка, но она обязана быть видна в отчёте.

    Ноль платежей связанной стороне за год — обычный исход, а вот ноль строк выручки
    почти наверняка означает промах классификации. Различить их здесь нечем, поэтому
    решение оставлено читателю отчёта.
    """
    return frozenset() if rows else frozenset({(Diagnostic.EMPTY_SELECTION, path)})


def _combine(amount: Decimal, parts: Sequence[Value]) -> Value:
    return Value(
        amount=amount,
        rows=frozenset().union(*(p.rows for p in parts)) if parts else frozenset(),
        facts=frozenset().union(*(p.facts for p in parts)) if parts else frozenset(),
        diagnostics=frozenset().union(*(p.diagnostics for p in parts)) if parts else frozenset(),
    )


__all__ = [
    "Diagnostic",
    "ExecutionError",
    "Executor",
    "Failure",
    "Operator",
    "Outcome",
    "TransactionCategory",
    "Value",
]
