"""Документальные приоритеты при подготовке реестра к исполнению."""

from datetime import date
from decimal import Decimal

from halyk.ingest.normalise import CovenantPeriod, NormalisedLedger
from halyk.knowledge.classifier import ClassificationResult
from halyk.models.adjustment import NormalisedTransaction
from halyk.models.classification import ClassificationRecord, DecisionSource, TransactionCategory
from halyk.models.fact import OneOffItemFact
from halyk.models.source import SourceRef
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, Money
from halyk.pipeline.borrower import with_categories


def transaction() -> NormalisedTransaction:
    amount = Money.from_decimal(Decimal("-342905.28"), Currency.USD)
    row = LedgerRow(
        txn_id="T1",
        date=date(2025, 1, 1),
        account_id="ACC-1",
        counterparty="Aral Freight Arbitration Bureau LLP",
        description="Demurrage dispute settlement",
        amount=amount,
    )
    return NormalisedTransaction(row=row, amount=amount, effective_date=row.date)


def classified() -> ClassificationResult:
    return ClassificationResult(
        records=(
            ClassificationRecord(
                txn_id="T1",
                account_id="ACC-1",
                final_category=TransactionCategory.OTHER,
                decision_source=DecisionSource.MODEL_ONLY,
            ),
        ),
        usage={},
    )


def fact(amount: str = "342905.28") -> OneOffItemFact:
    return OneOffItemFact(
        account_id="ACC-1",
        source=SourceRef(file_hash="a" * 64, file_name="notes.pdf", page=1),
        description="demurrage",
        counterparty="Aral Freight Arbitration Bureau LLP",
        amount=Money.from_decimal(Decimal(amount), Currency.USD),
    )


def ledger() -> NormalisedLedger:
    return NormalisedLedger(
        transactions=(transaction(),),
        adjustments=(),
        period=CovenantPeriod(date(2025, 1, 1), date(2025, 12, 31)),
    )


def test_documented_one_off_expense_overrides_model_category() -> None:
    result = with_categories(ledger(), classified(), (fact(),))
    assert result.transactions[0].covenant_category == TransactionCategory.OPEX.value


def test_one_off_fact_with_another_amount_does_not_override() -> None:
    result = with_categories(ledger(), classified(), (fact("1.00"),))
    assert result.transactions[0].covenant_category == TransactionCategory.OTHER.value
