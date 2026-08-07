"""Право документа менять расчёт."""

from halyk.knowledge.authority import (
    authoritative_report,
    final_reports,
    referenced_report,
    resolve_authority,
    superseded_drafts,
)
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus
from halyk.models.source import SourceAuthority

HASH = "0" * 64


def document(
    name: str,
    kind: DocumentKind,
    status: DocumentStatus,
    account_id: str | None = None,
    report_number: str | None = None,
) -> DocumentFacts:
    return DocumentFacts(
        file_name=name,
        sha256=HASH,
        kind=kind,
        status=status,
        account_id=account_id,
        report_number=report_number,
    )


def test_superseded_edition_cannot_change_the_answer() -> None:
    old = document("old.pdf", DocumentKind.LOAN_AGREEMENT, DocumentStatus.SUPERSEDED, "ACC-7201")
    assert resolve_authority(old) is SourceAuthority.IGNORED


def test_draft_is_recorded_but_not_authoritative() -> None:
    draft = document("d.pdf", DocumentKind.AUDIT_PROCEDURES, DocumentStatus.DRAFT, "ACC-7201")
    assert resolve_authority(draft) is SourceAuthority.RECORDED


def test_current_and_final_documents_are_authoritative() -> None:
    for kind, status in (
        (DocumentKind.LOAN_AGREEMENT, DocumentStatus.CURRENT),
        (DocumentKind.FINANCIAL_NOTES, DocumentStatus.CURRENT),
        (DocumentKind.KYC_FILE, DocumentStatus.CURRENT),
        (DocumentKind.TREASURY_MEMO, DocumentStatus.CURRENT),
        (DocumentKind.AUDIT_PROCEDURES, DocumentStatus.FINAL),
    ):
        assert resolve_authority(document("x.pdf", kind, status, "ACC-7201")) is (
            SourceAuthority.AUTHORITATIVE
        )


def test_unrelated_document_is_ignored() -> None:
    booklet = document("b.pdf", DocumentKind.UNRELATED, DocumentStatus.CURRENT)
    assert resolve_authority(booklet) is SourceAuthority.IGNORED


def test_report_number_alone_links_the_wrong_borrower() -> None:
    """Один и тот же номер стоит у финального отчёта одного заёмщика и у черновика другого.

    Это ровно та ловушка, ради которой отчёт ищется парой «счёт + номер».
    """
    documents = [
        document(
            "final.pdf",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.FINAL,
            "ACC-7201",
            "AR-2025-0634",
        ),
        document(
            "draft.pdf",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.DRAFT,
            "ACC-7803",
            "AR-2025-0634",
        ),
    ]
    assert set(final_reports(documents)) == {("ACC-7201", "AR-2025-0634")}

    text = "Вывод изложен в отчёте о выполнении согласованных процедур № AR-2025-0634."
    assert authoritative_report("ACC-7201", text, documents) is not None
    assert authoritative_report("ACC-7803", text, documents) is None


def test_reference_is_read_from_the_notes() -> None:
    text = (
        "Вывод по данной сумме изложен в отчёте о выполнении согласованных процедур № AR-2025-0634"
    )
    assert referenced_report(text) == "AR-2025-0634"
    assert referenced_report("Переклассификаций не требовалось") is None


def test_draft_is_superseded_only_when_its_borrower_has_a_final() -> None:
    documents = [
        document(
            "b1-draft.pdf",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.DRAFT,
            "ACC-7201",
            "AR-2025-7031",
        ),
        document(
            "b1-final.pdf",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.FINAL,
            "ACC-7201",
            "AR-2025-0634",
        ),
        document(
            "p3-draft.pdf",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.DRAFT,
            "ACC-7803",
            "AR-2025-0634",
        ),
    ]
    assert superseded_drafts(documents) == frozenset({"b1-draft.pdf"})
