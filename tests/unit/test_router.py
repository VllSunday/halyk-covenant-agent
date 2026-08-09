import pytest

from halyk.knowledge.router import (
    detect_account,
    detect_kind,
    detect_report_number,
    detect_status,
    squeeze,
)
from halyk.models.document import DocumentKind, DocumentStatus


def test_squeeze_survives_letter_spacing_and_broken_words() -> None:
    # Так заголовки выглядят в извлечённом слое этого датасета.
    assert squeeze("Д О Г О В О Р  Б А Н К О В С К О Г О  З А Й М А") == "ДОГОВОРБАНКОВСКОГОЗАЙМА"
    assert squeeze("КОНФИДЕНЦИА ЛЬНО") == "КОНФИДЕНЦИАЛЬНО"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Д О Г О В О Р Б А Н К О В С К О Г О З А Й М А", DocumentKind.LOAN_AGREEMENT),
        ("Отчёт о выполнении согласованных процедур", DocumentKind.AUDIT_PROCEDURES),
        ("Промежуточная ведомость вопросов по классификации", DocumentKind.AUDIT_PROCEDURES),
        ("Примечания к финансовой отчётности", DocumentKind.FINANCIAL_NOTES),
        ("Досье «Знай своего клиента» (KYC)", DocumentKind.KYC_FILE),
        ("Служебная записка казначейства", DocumentKind.TREASURY_MEMO),
        ("Руководство по бренду и логотипу", DocumentKind.UNRELATED),
    ],
)
def test_detect_kind(text: str, expected: DocumentKind) -> None:
    assert detect_kind(squeeze(text)) is expected


def test_superseded_edition_is_detected() -> None:
    text = "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.). ДОГОВОР БАНКОВСКОГО ЗАЙМА"
    assert detect_status(squeeze(text), DocumentKind.LOAN_AGREEMENT) is DocumentStatus.SUPERSEDED


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА", DocumentStatus.DRAFT),
        ("являются окончательной позицией аудитора", DocumentStatus.FINAL),
        ("обычный текст отчёта", DocumentStatus.CURRENT),
    ],
)
def test_audit_report_status(text: str, expected: DocumentStatus) -> None:
    assert detect_status(squeeze(text), DocumentKind.AUDIT_PROCEDURES) is expected


def test_draft_markers_do_not_apply_to_other_kinds() -> None:
    # Оговорка про неокончательность встречается и в служебных записках.
    text = squeeze("не является окончательной позицией")
    assert detect_status(text, DocumentKind.TREASURY_MEMO) is DocumentStatus.CURRENT


def test_account_is_the_most_mentioned() -> None:
    text = "ЗАЁМ № ACC-7801 ... счёт ACC-7801 ... сравнить с ACC-9999 ... ACC-7801"
    assert detect_account(text) == "ACC-7801"


def test_account_ties_resolve_to_the_first_mention() -> None:
    assert detect_account("ACC-7801 затем ACC-7802") == "ACC-7801"


def test_no_account() -> None:
    assert detect_account("документ без номеров счетов") is None


def test_report_number() -> None:
    assert detect_report_number("Номер заключения AR-2025-0634 выдан") == "AR-2025-0634"
    assert detect_report_number("без номера") is None


def test_account_is_matched_against_ledger_not_by_prefix() -> None:
    """Префикс счёта задаёт реестр, а не наше представление о его форме.

    Заёмщик со счётом вида `TELE-4471` иначе остаётся без единого документа, а
    номер отчёта `AR-2025` выигрывает у настоящего счёта по частоте упоминаний.
    """
    text = "ЗАЁМ № TELE-4471 ... досье AR-2025 ... AR-2025 ... TELE-4471 ... TELE-4471"
    assert detect_account(text, {"TELE-4471", "ACC-7001"}) == "TELE-4471"
    assert detect_account(text) is None


def test_credit_agreement_counts_only_in_the_heading() -> None:
    """Заголовок объявляет тип документа, ссылка в теле — нет."""
    agreement = squeeze(
        "HALYK BANK JSC CONFIDENTIAL · EXECUTION COPY · LOAN REFERENCE ACC-7604 "
        "CREDIT AGREEMENT Senior Secured Credit Facility"
    )
    assert detect_kind(agreement) is DocumentKind.LOAN_AGREEMENT

    notes = squeeze(
        "Notes to the Financial Statements "
        + "Note 1 — Basis of preparation. " * 20
        + "prepared solely for covenant-testing purposes under the Credit Agreement."
    )
    assert detect_kind(notes) is DocumentKind.FINANCIAL_NOTES
