"""Сбор фактов и корректировок по документам заёмщика."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from halyk.ingest.inventory import DatasetInventory
from halyk.ingest.scenarios import ScenarioMap
from halyk.knowledge.facts import build_facts
from halyk.models.adjustment import AdjustmentAction, AdjustmentStatus
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.models.fact import FactKind, OneOffItemFact, RelatedPartyPolicyFact
from halyk.models.source import SourceAuthority
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, Money

ACCOUNT = "ACC-7201"

FINAL_REPORT = """4. Выводы по классификации операций
(4.1) Сумма в размере $592,296.10, выплаченная контрагенту Irtysh Advisory Bureau,
первоначально учтённая как Консультационные услуги, переклассифицирована для целей
соблюдения ковенантов как Процентные расходы.
Основание: Вознаграждение по существу является платой за финансирование.
"""

DRAFT_REPORT = """4. Предварительные вопросы по классификации операций
(4.1) Операция TXN-B1-0023, первоначально учтённая как Операционные расходы
($6,166,592.66), переклассифицирована для целей соблюдения ковенантов как Коммунальные
услуги.
Основание: вопрос поставлен на промежуточном этапе.
"""

DOSSIER = """Организация
Доля голосующих прав
Ertis Capital, LLP
31.4%
Irtysh Advisory Bureau
19.2%
Организации, в которых Группа владеет 20.0% и более голосующих прав, признаются
связанными сторонами для целей Договора.
"""

ONE_OFF_NOTES = """Примечание 8 — Корректировки EBITDA

| Характер статьи | Контрагент | Сумма |
|---|---|---|
| Очистка причального дна | «Zhaiyk Dredging LLP» | $251,338.94 |

Разовыми для целей ковенантов признаются статьи в сумме не менее $300,000.00.
"""


def document(
    name: str,
    kind: DocumentKind,
    status: DocumentStatus,
    text: str = "",
    report_number: str | None = None,
    *,
    account_id: str | None = ACCOUNT,
) -> DocumentFacts:
    return DocumentFacts(
        file_name=name,
        sha256=name.encode().hex().ljust(64, "0")[:64],
        kind=kind,
        status=status,
        account_id=account_id,
        report_number=report_number,
        pages=(PageFacts(number=1, char_count=len(text), text=text),),
    )


def inventory(*documents: DocumentFacts) -> DatasetInventory:
    rows = (
        LedgerRow(
            txn_id="TXN-B1-0020",
            date=date(2025, 5, 12),
            account_id=ACCOUNT,
            counterparty="Irtysh Advisory Bureau",
            description="Advisory engagement",
            amount=Money.from_decimal(Decimal("-592296.10"), Currency.USD),
        ),
    )
    return DatasetInventory(
        documents=documents,
        ledger=rows,
        scenarios=ScenarioMap.build(["B1"], rows),
        ledger_path=Path("ledger.csv"),
    )


def test_final_report_produces_an_adjustment_waiting_for_the_ledger() -> None:
    found = build_facts(
        inventory(
            document(
                "final.pdf",
                DocumentKind.AUDIT_PROCEDURES,
                DocumentStatus.FINAL,
                FINAL_REPORT,
                "AR-2025-0634",
            )
        )
    )
    assert len(found.adjustments) == 1
    change = found.adjustments[0]
    assert change.action is AdjustmentAction.RECLASSIFY
    assert change.status is AdjustmentStatus.PENDING
    assert change.source.authority is SourceAuthority.AUTHORITATIVE
    assert change.selector.counterparty == "Irtysh Advisory Bureau"


def test_draft_is_superseded_when_the_borrower_has_a_final() -> None:
    found = build_facts(
        inventory(
            document(
                "draft.pdf",
                DocumentKind.AUDIT_PROCEDURES,
                DocumentStatus.DRAFT,
                DRAFT_REPORT,
                "AR-2025-7031",
            ),
            document(
                "final.pdf",
                DocumentKind.AUDIT_PROCEDURES,
                DocumentStatus.FINAL,
                FINAL_REPORT,
                "AR-2025-0634",
            ),
        )
    )
    statuses = {a.source.file_name: a.status for a in found.adjustments}
    assert statuses["draft.pdf"] is AdjustmentStatus.SUPERSEDED
    assert statuses["final.pdf"] is AdjustmentStatus.PENDING


def test_draft_without_a_final_stays_unconfirmed() -> None:
    """Ведомость без окончательного отчёта не меняет первоначальную классификацию."""
    found = build_facts(
        inventory(
            document(
                "draft.pdf",
                DocumentKind.AUDIT_PROCEDURES,
                DocumentStatus.DRAFT,
                DRAFT_REPORT,
                "AR-2025-7031",
            )
        )
    )
    assert found.adjustments[0].status is AdjustmentStatus.UNCONFIRMED


def test_superseded_agreement_contributes_nothing() -> None:
    found = build_facts(
        inventory(
            document(
                "old.pdf", DocumentKind.LOAN_AGREEMENT, DocumentStatus.SUPERSEDED, FINAL_REPORT
            )
        )
    )
    assert found.facts == ()
    assert found.adjustments == ()


def test_dossier_facts_carry_their_address() -> None:
    dossier = document("kyc.pdf", DocumentKind.KYC_FILE, DocumentStatus.CURRENT, DOSSIER)
    found = build_facts(inventory(dossier))
    fact = found.facts[0]
    assert isinstance(fact, RelatedPartyPolicyFact)
    assert fact.threshold == Decimal("0.20")
    assert fact.related_parties == ("Ertis Capital, LLP",)
    assert (fact.source.file_name, fact.source.page) == ("kyc.pdf", 1)
    assert fact.source.file_hash == dossier.sha256
    assert fact.source.kind is DocumentKind.KYC_FILE


def test_one_off_items_are_facts_and_not_transactions() -> None:
    """Разовая статья не проводка: синтетическая операция сломала бы объяснение ответа."""
    found = build_facts(
        inventory(
            document(
                "notes.pdf", DocumentKind.FINANCIAL_NOTES, DocumentStatus.CURRENT, ONE_OFF_NOTES
            )
        )
    )
    kinds = [fact.kind for fact in found.facts]
    assert kinds == [FactKind.ONE_OFF_ITEM, FactKind.ONE_OFF_POLICY]
    assert found.adjustments == ()
    item = found.facts[0]
    assert isinstance(item, OneOffItemFact)
    assert item.amount == Money.from_decimal(Decimal("251338.94"), Currency.USD)


def test_unparsed_dossier_is_reported() -> None:
    blank = document("kyc.pdf", DocumentKind.KYC_FILE, DocumentStatus.CURRENT, "Пустое досье")
    assert build_facts(inventory(blank)).unparsed_dossiers == ("kyc.pdf",)


@pytest.mark.parametrize("account_id", [None, ""])
def test_document_without_an_account_is_skipped(account_id: str | None) -> None:
    orphan = document(
        "notes.pdf",
        DocumentKind.FINANCIAL_NOTES,
        DocumentStatus.CURRENT,
        ONE_OFF_NOTES,
        account_id=account_id,
    )
    assert build_facts(inventory(orphan)).facts == ()
