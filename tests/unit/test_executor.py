"""Вычислительная семантика исполнителя."""

from __future__ import annotations

import random
from datetime import date
from decimal import Decimal

import pytest

from halyk.execution.executor import Diagnostic, Executor, Failure
from halyk.knowledge.kyc import RelatedPartyPolicy
from halyk.knowledge.related_party import build_index
from halyk.models.adjustment import NormalisedTransaction
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, RoundingSpec, Unit
from halyk.models.fact import OneOffItemFact, OneOffPolicyFact
from halyk.models.formula import (
    Condition,
    Constant,
    CovenantFormula,
    Difference,
    Direction,
    EvidenceMode,
    ExternalMetric,
    FactValue,
    Largest,
    LedgerSum,
    Ratio,
    RelatedParty,
    Selector,
    Sum,
)
from halyk.models.source import SourceRef
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, Money

ACCOUNT = "ACC-7801"
SOURCE = SourceRef(file_hash="a" * 64, file_name="doc.pdf", page=1)


def txn(
    txn_id: str,
    amount: str,
    category: Cat,
    *,
    counterparty: str = "Northwind Catering",
    month: int = 3,
) -> NormalisedTransaction:
    money = Money.from_decimal(Decimal(amount), Currency.USD)
    return NormalisedTransaction(
        row=LedgerRow(
            txn_id=txn_id,
            date=date(2025, month, 1),
            account_id=ACCOUNT,
            counterparty=counterparty,
            description="строка реестра",
            amount=money,
        ),
        amount=money,
        effective_date=date(2025, month, 1),
        covenant_category=category.value,
    )


def executor(*transactions: NormalisedTransaction, **kwargs: object) -> Executor:
    return Executor(account_id=ACCOUNT, transactions=transactions, facts=(), **kwargs)  # type: ignore[arg-type]


def formula(**overrides: object) -> CovenantFormula:
    base: dict[str, object] = {
        "scenario_id": "P1",
        "clause_id": "6.1",
        "title": "проверка",
        "measure": LedgerSum(selector=Selector(categories=(Cat.CAPEX,))),
        "operator": Operator.LE,
        "threshold": Constant(value=Decimal("300000")),
        "unit": Unit.MONEY,
        "source_refs": (SOURCE,),
        "quote": "не более 300 000",
        "confidence": 0.9,
    }
    return CovenantFormula(**(base | overrides))  # type: ignore[arg-type]


def spend(*categories: Cat) -> LedgerSum:
    return LedgerSum(selector=Selector(categories=categories, direction=Direction.OUTFLOW))


# --- базовая арифметика ------------------------------------------------------


def test_sum_takes_the_absolute_value() -> None:
    """Расходы в реестре отрицательны, а actual по условию всегда положителен."""
    result = executor(txn("T1", "-1000.50", Cat.CAPEX), txn("T2", "-2000.25", Cat.CAPEX)).run(
        formula()
    )
    assert result.actual == Decimal("3000.75")
    assert result.status == "COMPLIANT"


def test_arithmetic_stays_in_decimal() -> None:
    """Копейки на суммах в миллионы: float потерял бы хвост, Decimal — нет."""
    result = executor(txn("T1", "-10000000.01", Cat.CAPEX), txn("T2", "-0.02", Cat.CAPEX)).run(
        formula(threshold=Constant(value=Decimal("99999999")))
    )
    assert result.actual == Decimal("10000000.03")


def test_ratio_and_rounding() -> None:
    result = executor(
        txn("T1", "-4000", Cat.CAPEX),
        txn("T2", "-3000", Cat.OPEX),
    ).run(
        formula(
            measure=Ratio(numerator=spend(Cat.CAPEX), denominator=spend(Cat.OPEX)),
            unit=Unit.RATIO,
            threshold=Constant(value=Decimal("2")),
            rounding=RoundingSpec(scale=4),
        )
    )
    assert result.actual == Decimal("1.3333")


def test_ebitda_with_add_back_above_the_policy() -> None:
    """Разовая статья ниже порога существенности в EBITDA не попадает."""
    facts = (
        OneOffPolicyFact(
            account_id=ACCOUNT, source=SOURCE, minimum=Money.from_decimal(300000, Currency.USD)
        ),
        OneOffItemFact(
            account_id=ACCOUNT,
            source=SOURCE,
            description="крупная разовая статья",
            counterparty="X",
            amount=Money.from_decimal(500000, Currency.USD),
        ),
        OneOffItemFact(
            account_id=ACCOUNT,
            source=SOURCE,
            description="мелкая разовая статья",
            counterparty="Y",
            amount=Money.from_decimal(1000, Currency.USD),
        ),
    )
    engine = Executor(
        account_id=ACCOUNT,
        transactions=(txn("T1", "3000000", Cat.REVENUE), txn("T2", "-1000000", Cat.OPEX)),
        facts=facts,
    )
    result = engine.run(
        formula(
            measure=Sum(
                terms=(
                    Difference(
                        left=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
                        right=spend(Cat.OPEX),
                    ),
                    FactValue(fact_kind="one_off_item", above_one_off_policy=True),
                )
            ),
            threshold=Constant(value=Decimal("99999999")),
        )
    )
    assert result.actual == Decimal("2500000.00")


def test_largest_of_two_caps() -> None:
    """«По отдельности, а не в совокупности»: потолок меряет наибольшую статью."""
    result = executor(
        txn("T1", "-900", Cat.PAYROLL),
        txn("T2", "-1200", Cat.UTILITIES),
    ).run(
        formula(
            measure=Largest(values=(spend(Cat.PAYROLL), spend(Cat.UTILITIES))),
            threshold=Constant(value=Decimal("1000")),
        )
    )
    assert result.actual == Decimal("1200.00")
    assert result.status == "BREACH"


def test_months_cut_the_period() -> None:
    result = executor(
        txn("T1", "500", Cat.REVENUE, month=3),
        txn("T2", "700", Cat.REVENUE, month=11),
    ).run(
        formula(
            measure=LedgerSum(selector=Selector(categories=(Cat.REVENUE,), months=(10, 11, 12))),
            operator=Operator.GE,
            threshold=Constant(value=Decimal("600")),
        )
    )
    assert result.actual == Decimal("700.00")
    assert result.rows == ("T2",)


# --- отказы ------------------------------------------------------------------


def test_external_metric_stops_the_cell() -> None:
    result = executor(txn("T1", "-1000", Cat.CAPEX)).run(
        formula(
            measure=Ratio(
                numerator=spend(Cat.CAPEX),
                denominator=ExternalMetric(name="group_capex", description="не раскрыта"),
            ),
            unit=Unit.RATIO,
        )
    )
    assert result.failure is Failure.MISSING_EXTERNAL_METRIC
    assert result.failure_path == "measure.denominator"
    assert result.actual is None and result.status is None


def test_zero_denominator_is_typed() -> None:
    result = executor(txn("T1", "-1000", Cat.CAPEX)).run(
        formula(
            measure=Ratio(numerator=spend(Cat.CAPEX), denominator=spend(Cat.OPEX)),
            unit=Unit.RATIO,
        )
    )
    assert result.failure is Failure.ZERO_DENOMINATOR


def test_related_party_filter_without_a_dossier_is_typed() -> None:
    result = executor(txn("T1", "-1000", Cat.OPEX)).run(
        formula(
            measure=LedgerSum(
                selector=Selector(categories=(Cat.OPEX,), related_party=RelatedParty.ONLY)
            )
        )
    )
    assert result.failure is Failure.MISSING_KYC_POLICY


def test_ambiguous_counterparty_is_typed() -> None:
    index = build_index(
        ACCOUNT,
        RelatedPartyPolicy(
            threshold=Decimal("0.20"), holdings=(("Aktau Holdings LLP", Decimal("0.35")),)
        ),
    )
    result = executor(
        txn("T1", "-1000", Cat.OPEX, counterparty="Aktau Holdinqs LLP"), related=index
    ).run(
        formula(
            measure=LedgerSum(
                selector=Selector(categories=(Cat.OPEX,), related_party=RelatedParty.ONLY)
            )
        )
    )
    assert result.failure is Failure.AMBIGUOUS_COUNTERPARTY


def test_unresolved_counterparty_outside_the_selector_is_harmless() -> None:
    """Строка, отсеянная по категории, не должна ронять прогон своим контрагентом."""
    index = build_index(
        ACCOUNT,
        RelatedPartyPolicy(
            threshold=Decimal("0.20"), holdings=(("Aktau Holdings LLP", Decimal("0.35")),)
        ),
    )
    result = executor(
        txn("T1", "-1000", Cat.CAPEX, counterparty="Aktau Holdinqs LLP"),
        txn("T2", "-500", Cat.OPEX, counterparty="Aktau Holdings L.L.P."),
        related=index,
    ).run(
        formula(
            measure=LedgerSum(
                selector=Selector(categories=(Cat.OPEX,), related_party=RelatedParty.ONLY)
            )
        )
    )
    assert result.is_resolved
    assert result.actual == Decimal("500.00")


def test_related_party_without_transactions_gives_zero() -> None:
    index = build_index(
        ACCOUNT,
        RelatedPartyPolicy(
            threshold=Decimal("0.20"), holdings=(("Aktau Holdings LLP", Decimal("0.35")),)
        ),
    )
    result = executor(txn("T1", "-1000", Cat.OPEX), related=index).run(
        formula(
            measure=LedgerSum(
                selector=Selector(categories=(Cat.OPEX,), related_party=RelatedParty.ONLY)
            )
        )
    )
    assert result.actual == Decimal("0.00")
    assert (Diagnostic.EMPTY_SELECTION, "measure") in result.diagnostics


def test_only_and_exclude_partition_the_rows() -> None:
    index = build_index(
        ACCOUNT,
        RelatedPartyPolicy(
            threshold=Decimal("0.20"), holdings=(("Aktau Holdings LLP", Decimal("0.35")),)
        ),
    )
    rows = (
        txn("T1", "-1000", Cat.OPEX, counterparty="Aktau Holdings L.L.P."),
        txn("T2", "-500", Cat.OPEX, counterparty="Northwind Catering"),
    )

    def total(mode: RelatedParty) -> Decimal | None:
        return (
            executor(*rows, related=index)
            .run(
                formula(
                    measure=LedgerSum(
                        selector=Selector(categories=(Cat.OPEX,), related_party=mode)
                    ),
                    threshold=Constant(value=Decimal("99999")),
                )
            )
            .actual
        )

    assert total(RelatedParty.ONLY) == Decimal("1000.00")
    assert total(RelatedParty.EXCLUDE) == Decimal("500.00")
    assert total(RelatedParty.ANY) == Decimal("1500.00")


# --- условные пункты ---------------------------------------------------------


def test_untriggered_covenant_is_compliant_but_keeps_its_measure() -> None:
    """Пока условие не сработало, нарушения нет, а фактическое значение всё равно есть."""
    result = executor(
        txn("T1", "-5000", Cat.CAPEX),
        txn("T2", "-100", Cat.INTEREST_EXPENSE),
    ).run(
        formula(
            threshold=Constant(value=Decimal("1000")),
            applies_when=Condition(
                left=spend(Cat.INTEREST_EXPENSE),
                operator=Operator.GT,
                right=Constant(value=Decimal("1000000")),
            ),
        )
    )
    assert result.status == "COMPLIANT"
    assert result.triggered is False
    assert result.actual == Decimal("5000.00")


# --- улика -------------------------------------------------------------------


def test_evidence_is_the_row_whose_removal_flips_the_verdict() -> None:
    """Порог 950 при сумме 1100: снятие любой мелкой строки оставляет нарушение."""
    result = executor(
        txn("T1", "-100", Cat.CAPEX),
        txn("T2", "-100", Cat.CAPEX),
        txn("T3", "-900", Cat.CAPEX),
    ).run(formula(threshold=Constant(value=Decimal("950"))))
    assert result.status == "BREACH"
    assert result.evidence_txn_id == "T3"


def test_contributing_row_is_not_evidence() -> None:
    """Вклад в сумму уликой не делает — условие задачи прямо это отвергает."""
    result = executor(
        txn("T1", "-600", Cat.CAPEX),
        txn("T2", "-600", Cat.CAPEX),
    ).run(formula(threshold=Constant(value=Decimal("1000"))))
    assert result.status == "BREACH"
    # Изъятие любой из двух переворачивает вердикт, единственной причины нет.
    assert result.evidence_txn_id is None


def test_no_evidence_when_nothing_flips_the_verdict() -> None:
    result = executor(
        txn("T1", "-5000", Cat.CAPEX),
        txn("T2", "-5000", Cat.CAPEX),
    ).run(formula(threshold=Constant(value=Decimal("1000"))))
    assert result.status == "BREACH"
    assert result.evidence_txn_id is None


def test_compliant_cell_has_no_evidence() -> None:
    result = executor(txn("T1", "-100", Cat.CAPEX)).run(formula())
    assert result.status == "COMPLIANT"
    assert result.evidence_txn_id is None


def test_evidence_is_suppressed_when_the_mode_says_so() -> None:
    result = executor(
        txn("T1", "-100", Cat.CAPEX),
        txn("T2", "-900", Cat.CAPEX),
    ).run(formula(threshold=Constant(value=Decimal("500")), evidence=EvidenceMode.NONE))
    assert result.status == "BREACH"
    assert result.evidence_txn_id is None


# --- метаморфные -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_permutation_changes_nothing(seed: int) -> None:
    """Перестановка строк не меняет ни числа, ни вердикта, ни улики."""
    rows = [
        txn("T1", "-100.11", Cat.CAPEX),
        txn("T2", "-200.22", Cat.CAPEX),
        txn("T3", "-900.33", Cat.CAPEX),
        txn("T4", "-50.44", Cat.OPEX),
        txn("T5", "1500.55", Cat.REVENUE),
    ]
    spec = formula(threshold=Constant(value=Decimal("1000")))
    reference = executor(*rows).run(spec)

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    result = executor(*shuffled).run(spec)

    assert (result.actual, result.status, result.evidence_txn_id) == (
        reference.actual,
        reference.status,
        reference.evidence_txn_id,
    )
    assert result.rows == reference.rows


def test_ratio_is_stable_under_permutation() -> None:
    rows = [txn(f"T{i}", f"-{i}00.07", Cat.OPEX) for i in range(1, 8)]
    rows.append(txn("R1", "9999.13", Cat.REVENUE))
    spec = formula(
        measure=Ratio(
            numerator=LedgerSum(selector=Selector(categories=(Cat.REVENUE,))),
            denominator=spend(Cat.OPEX),
        ),
        unit=Unit.RATIO,
        rounding=RoundingSpec(scale=6),
        threshold=Constant(value=Decimal("1")),
    )
    first = executor(*rows).run(spec)
    second = executor(*reversed(rows)).run(spec)
    assert first.actual == second.actual
