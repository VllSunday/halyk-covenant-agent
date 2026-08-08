"""Полностью англоязычный набор проходит весь путь до вердикта.

Матрица в `test_bilingual` проверяет семантики поодиночке. Здесь проверяется, что они
складываются: если хоть одно звено осталось русскоязычным, ячейка не посчитается или
посчитается неверно, а прогон при этом останется зелёным.

Набор синтетический и маленький: PDF собираются на месте, API не вызывается.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import fitz
import pytest

from halyk.execution.executor import Executor
from halyk.ingest.inventory import build_inventory
from halyk.ingest.normalise import CovenantPeriod, normalise
from halyk.knowledge.facts import build_facts
from halyk.knowledge.related_party import build_index
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.document import DocumentKind, DocumentStatus
from halyk.models.fact import FactKind, RelatedPartyPolicyFact
from halyk.models.formula import (
    Constant,
    CovenantFormula,
    Direction,
    LedgerSum,
    RelatedParty,
    Selector,
)
from halyk.models.source import SourceRef

PERIOD = CovenantPeriod(start=date(2025, 1, 1), end=date(2025, 12, 31))

AGREEMENT = """BANK LOAN AGREEMENT
Borrower: Aktau Energy JSC
Account ACC-4001

Article 6 — Financial covenants
Clause 6.1 Maximum payments to related parties. Over the period from 2025-01-01 to
2025-12-31 the aggregate payments of the Borrower to affiliated and related parties
shall not exceed $200,000.00. Account ACC-4001.
"""

SUPERSEDED_AGREEMENT = """BANK LOAN AGREEMENT
SUPERSEDED EDITION (2024). Replaced by a new edition.
Borrower: Aktau Energy JSC
Account ACC-4001
Clause 6.1 Maximum payments to related parties shall not exceed $50,000.00.
Account ACC-4001.
"""

KYC = """KNOW YOUR CUSTOMER FILE
Related party review

Entity
Aktau Energy JSC
Account
ACC-4001

Share of voting rights
Sarybel Capital LLP
41.2%
Pavlodar Plant Services LLP
18.0%
Entities in which the Group holds 20.0% or more of voting rights are treated as
related parties for the purposes of the Agreement.
Account ACC-4001
"""

NOTES = """NOTES TO THE FINANCIAL STATEMENTS
Account ACC-4001

(9.1) Transaction TXN-E1-0002, originally recorded as Other expenses ($120,000.00),
has been reclassified for covenant purposes as Operating costs. Basis: the agreement.

(9.2) Transaction TXN-E1-0004, dated 2025-02-01, is excluded from the covenant period.

(9.3) Items of not less than $300,000.00 are treated as one-off for covenant purposes.
"""

LEDGER = """txn_id,date,account_id,counterparty,description,amount,currency
TXN-E1-0001,2025-03-01,ACC-4001,Sarybel Capital L.L.P.,Advisory services,-150000.00,USD
TXN-E1-0002,2025-04-01,ACC-4001,Sarybel Capital LLP,Site servicing costs,-120000.00,USD
TXN-E1-0003,2025-05-01,ACC-4001,Northwind Catering,Catering for the depot,-90000.00,USD
TXN-E1-0004,2025-02-01,ACC-4001,Sarybel Capital LLP,Advance excluded by the auditor,-500000.00,USD
"""

TEMPLATE = """{
  "team": "", "contact_email": "", "model": "",
  "answers": {"E1": {"6.1": {"status": "", "actual": null, "evidence_txn_id": null}}}
}
"""


def write_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    for chunk in [text[i : i + 1800] for i in range(0, len(text), 1800)] or [""]:
        page = document.new_page()
        page.insert_text((40, 60), chunk, fontsize=9)
    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("english-dataset")
    (root / "documents").mkdir()
    write_pdf(root / "documents" / "agreement.pdf", AGREEMENT)
    write_pdf(root / "documents" / "superseded.pdf", SUPERSEDED_AGREEMENT)
    write_pdf(root / "documents" / "kyc.pdf", KYC)
    write_pdf(root / "documents" / "notes.pdf", NOTES)
    (root / "master_ledger_2025.csv").write_text(LEDGER, encoding="utf-8")
    (root / "submission_template.json").write_text(TEMPLATE, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def inventory(dataset: Path):  # type: ignore[no-untyped-def]
    return build_inventory(dataset, ["E1"])


def test_every_document_is_classified(inventory) -> None:  # type: ignore[no-untyped-def]
    kinds = {doc.file_name: doc.kind for doc in inventory.documents}
    assert kinds["agreement.pdf"] is DocumentKind.LOAN_AGREEMENT
    assert kinds["kyc.pdf"] is DocumentKind.KYC_FILE
    assert kinds["notes.pdf"] is DocumentKind.FINANCIAL_NOTES
    assert inventory.unresolved_documents == ()


def test_superseded_edition_is_separated(inventory) -> None:  # type: ignore[no-untyped-def]
    """Иначе у заёмщика окажется два действующих договора с разными порогами."""
    statuses = {doc.file_name: doc.status for doc in inventory.documents}
    assert statuses["agreement.pdf"] is DocumentStatus.CURRENT
    assert statuses["superseded.pdf"] is DocumentStatus.SUPERSEDED


def test_language_is_reported_as_english(inventory) -> None:  # type: ignore[no-untyped-def]
    assert inventory.report()["languages"]["en"] == 4


def test_kyc_and_disclosures_are_read(inventory) -> None:  # type: ignore[no-untyped-def]
    facts = build_facts(inventory)
    assert facts.unparsed_dossiers == ()
    assert facts.unparsed_disclosures == ()

    policy = next(f for f in facts.facts if f.kind is FactKind.RELATED_PARTY_POLICY)
    assert isinstance(policy, RelatedPartyPolicyFact)
    assert policy.threshold == Decimal("0.20")
    assert policy.related_parties == ("Sarybel Capital LLP",)

    actions = {a.action for a in facts.adjustments}
    assert actions >= {"reclassify", "exclude"}


def test_related_party_covenant_is_computed(inventory) -> None:  # type: ignore[no-untyped-def]
    """Сквозной путь: перепись, факты, нормализация, расчёт, вердикт.

    Ожидаемое число: платежи связанной стороне за период — 150 000 и 120 000. Строка
    на 500 000 исключена раскрытием, а 90 000 ушли несвязанному контрагенту. Итог
    270 000 против порога 200 000, то есть нарушение.
    """
    facts = build_facts(inventory)
    ledger = normalise(inventory.ledger, facts.adjustments, PERIOD)
    policy = next(f for f in facts.facts if f.kind is FactKind.RELATED_PARTY_POLICY)

    engine = Executor(
        account_id="ACC-4001",
        transactions=ledger.transactions,
        facts=facts.by_account("ACC-4001"),
        related=build_index("ACC-4001", policy.as_policy()),
    )
    result = engine.run(
        CovenantFormula(
            scenario_id="E1",
            clause_id="6.1",
            title="Maximum payments to related parties",
            measure=LedgerSum(
                selector=Selector(direction=Direction.OUTFLOW, related_party=RelatedParty.ONLY)
            ),
            operator=Operator.LE,
            threshold=Constant(value=Decimal("200000")),
            unit=Unit.MONEY,
            source_refs=(SourceRef(file_hash="a" * 64, file_name="agreement.pdf", page=1),),
            quote="shall not exceed $200,000.00",
            confidence=0.9,
        )
    )

    assert result.is_resolved, result.failure_detail
    assert result.actual == Decimal("270000.00")
    assert result.status == "BREACH"
    # `Sarybel Capital L.L.P.` из реестра сопоставлен с `Sarybel Capital LLP` из досье.
    assert "TXN-E1-0001" in result.rows
    assert "TXN-E1-0004" not in result.rows


def test_reclassification_reached_the_ledger(inventory) -> None:  # type: ignore[no-untyped-def]
    facts = build_facts(inventory)
    ledger = normalise(inventory.ledger, facts.adjustments, PERIOD)
    reclassified = ledger.by_id("TXN-E1-0002")
    assert reclassified is not None
    assert reclassified.covenant_category == Cat.OPEX.value or reclassified.is_adjusted
