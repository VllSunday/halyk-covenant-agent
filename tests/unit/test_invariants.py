"""Инварианты прогона.

Проверки должны молчать на исправных данных и говорить на испорченных. Второе
важнее: инвариант, который никогда не срабатывает, ничего не охраняет.
"""

from __future__ import annotations

from pathlib import Path

from halyk.ingest.inventory import DatasetInventory
from halyk.ingest.scenarios import ScenarioMap
from halyk.knowledge.facts import FactSet
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.verification.invariants import (
    Severity,
    check_coverage,
    check_documents,
    check_facts,
    relevance_signals,
    summarise,
)

MONEY_TABLE = "$1,000.00 $2,000.00 $3,000.00 $4,000.00"


def page(text: str) -> PageFacts:
    return PageFacts(number=1, text=text, char_count=len(text))


def document(
    name: str,
    *,
    kind: DocumentKind = DocumentKind.LOAN_AGREEMENT,
    status: DocumentStatus = DocumentStatus.CURRENT,
    account: str | None = "ACC-7801",
    text: str = "договор",
) -> DocumentFacts:
    return DocumentFacts(
        file_name=name,
        sha256="a" * 64,
        kind=kind,
        status=status,
        account_id=account,
        pages=(page(text),),
    )


def inventory(*documents: DocumentFacts, unresolved: tuple[str, ...] = ()) -> DatasetInventory:
    return DatasetInventory(
        documents=documents,
        ledger=(),
        scenarios=ScenarioMap(
            scenario_to_account={"P1": "ACC-7801"}, account_to_scenario={"ACC-7801": "P1"}
        ),
        ledger_path=Path("ledger.csv"),
        unresolved_documents=unresolved,
    )


def codes(violations: list) -> set[str]:  # type: ignore[type-arg]
    return {v.code for v in violations}


# --- тишина на исправных данных ----------------------------------------------


def test_healthy_inventory_is_silent() -> None:
    assert check_documents(inventory(document("agreement.pdf"))) == []


def test_noise_document_is_not_reported() -> None:
    """Брендбук с одной суммой в отчёт попадать не должен."""
    noise = document(
        "brand.pdf",
        kind=DocumentKind.UNRELATED,
        account=None,
        text="Руководство по бренду. Бюджет $1,000.00 на квартал. ACC-7801.",
    )
    assert check_documents(inventory(document("agreement.pdf"), noise)) == []


def test_compliance_procedure_is_not_reported() -> None:
    """Регламент «о периодическом обновлении KYC» — процесс, а не обязательство."""
    procedure = document(
        "procedure.pdf",
        kind=DocumentKind.UNRELATED,
        account=None,
        text="Процедура комплаенса — периодическое обновление KYC. Счёт ACC-7801.",
    )
    assert check_documents(inventory(document("agreement.pdf"), procedure)) == []


def test_short_amendment_with_one_amount_is_reported() -> None:
    """Одной денежной плотности мало: допсоглашение решает ячейку одной строкой."""
    amendment = document(
        "amendment.pdf",
        kind=DocumentKind.UNRELATED,
        account=None,
        text=(
            "Дополнительное соглашение № 2 к Договору банковского займа. "
            "Пункт 6.3 излагается в новой редакции: порог не более $450,000.00. "
            "Счёт ACC-7801."
        ),
    )
    found = check_documents(inventory(document("agreement.pdf"), amendment))
    assert codes(found) == {"unclassified_but_relevant"}
    assert "clause_number" in found[0].message


def test_signals_are_named_in_the_message() -> None:
    """По отчёту должно быть видно, чем документ похож на значимый."""
    assert set(relevance_signals(f"Пункт 6.1 covenant {MONEY_TABLE} ACC-7801")) == {
        "money",
        "clause_number",
        "account",
        "contract_vocabulary",
    }


# --- срабатывание на испорченных ---------------------------------------------


def test_unclassified_document_with_money_is_reported() -> None:
    """Так был потерян консолидированный отчёт: тип не распознан, документ в шуме.

    Признаки взяты с настоящего файла: денежная плотность и договорная лексика
    («audited»). Номера счёта в нём нет вовсе, поэтому по счёту он не нашёлся бы.
    """
    orphan = document(
        "report.pdf",
        kind=DocumentKind.UNRELATED,
        account=None,
        text=f"Consolidated Financial Statements, audited. {MONEY_TABLE}",
    )
    found = check_documents(inventory(document("agreement.pdf"), orphan))
    assert codes(found) == {"unclassified_but_relevant"}
    assert found[0].subject == "report.pdf"


def test_two_current_editions_are_reported() -> None:
    """Нераспознанный маркер вытеснения даёт верное число при неверном пороге."""
    found = check_documents(inventory(document("agreement.pdf"), document("second-edition.pdf")))
    assert codes(found) == {"edition_is_ambiguous"}
    assert found[0].severity is Severity.ERROR


def test_missing_agreement_is_reported() -> None:
    superseded = document("old.pdf", status=DocumentStatus.SUPERSEDED)
    found = check_documents(inventory(superseded))
    assert codes(found) == {"edition_is_ambiguous"}
    assert "нет ни одного" in found[0].message


def test_document_without_an_account_is_reported() -> None:
    found = check_documents(inventory(document("agreement.pdf"), unresolved=("mystery.pdf",)))
    assert codes(found) == {"document_without_account"}


# --- слой фактов --------------------------------------------------------------


def test_unparsed_dossier_and_disclosure_are_reported() -> None:
    facts = FactSet(
        unparsed_dossiers=("kyc.pdf",),
        unparsed_disclosures=("notes.pdf#9.1",),
    )
    found = check_facts(facts, inventory(document("agreement.pdf")))
    assert {"dossier_not_parsed", "disclosure_not_parsed"} <= codes(found)


def test_borrower_without_a_related_party_policy_is_a_warning() -> None:
    """Предупреждение, а не ошибка: не каждому ковенанту нужен этот порог."""
    found = check_facts(FactSet(), inventory(document("agreement.pdf")))
    policy = [v for v in found if v.code == "related_party_policy_missing"]
    assert len(policy) == 1
    assert policy[0].severity is Severity.WARNING


# --- покрытие ячеек -----------------------------------------------------------


def test_missing_cell_is_reported() -> None:
    found = check_coverage(["P1/6.1"], ["P1/6.1", "P1/6.2"])
    assert codes(found) == {"cell_not_answered"}
    assert found[0].subject == "P1/6.2"


def test_full_coverage_is_silent() -> None:
    assert check_coverage(["P1/6.1", "P1/6.2"], ["P1/6.1", "P1/6.2"]) == []


def test_summary_separates_errors_from_warnings() -> None:
    found = check_facts(FactSet(unparsed_dossiers=("kyc.pdf",)), inventory(document("a.pdf")))
    summary = summarise(found)
    assert summary["errors"] == 1
    assert summary["warnings"] == 1
