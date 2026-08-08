"""Определение типа, принадлежности и статуса документа по его содержимому.

Имена файлов обезличены, поэтому единственный источник — текст. Правила здесь
наблюдённые, а не взятые из условия задачи, поэтому каждое из них может не
сработать на другом наборе: нераспознанное не подменяется догадкой, а попадает в
отчёт отдельным списком.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from halyk.knowledge.kyc import normalise_counterparty
from halyk.models.document import DocumentKind, DocumentStatus

ACCOUNT_PATTERN = re.compile(r"ACC-\d{4}")
REPORT_PATTERN = re.compile(r"\bAR-\d{4}-\d{4}\b")
_WHITESPACE = re.compile(r"\s+")

# Досье объявляет владельца счёта парой строк «Организация» и «Счёт». Метки идут на
# обоих языках: в приватном наборе досье может прийти на английском.
_ORGANISATION = re.compile(
    r"(?:Организация|Organisation|Organization|Entity)\s*\n\s*(?P<name>\S[^\n]*?)\s*\n"
    r"\s*(?:Счёт|Счет|Account)\s*\n\s*ACC-\d{4}",
    re.IGNORECASE,
)

# Маркеры сравниваются с текстом, из которого убраны все пробелы. В извлечённом
# слое заголовки идут вразрядку («Д О Г О В О Р») и рвутся посреди слова
# («КОНФИДЕНЦИА ЛЬНО»), и обычный поиск подстроки на них не работает.
# Маркеры перечислены на обоих языках. Организаторы предупредили, что документы могут
# быть на английском, и в открытом наборе такой ровно один — консолидированная
# отчётность материнской компании. Она относится к заёмщику, но по русским маркерам не
# опознавалась и молча уходила в шум вместе с величиной, без которой не считается
# ковенант P5/6.1.
_KIND_MARKERS: tuple[tuple[DocumentKind, tuple[str, ...]], ...] = (
    (
        DocumentKind.LOAN_AGREEMENT,
        ("ДОГОВОРБАНКОВСКОГОЗАЙМА", "BANKLOANAGREEMENT", "LOANAGREEMENT"),
    ),
    (
        DocumentKind.AUDIT_PROCEDURES,
        (
            "ОТЧЁТОВЫПОЛНЕНИИСОГЛАСОВАННЫХПРОЦЕДУР",
            "ВЕДОМОСТЬВОПРОСОВПОКЛАССИФИКАЦИИ",
            "AGREEDUPONPROCEDURES",
            "CLASSIFICATIONQUERYSCHEDULE",
        ),
    ),
    (
        DocumentKind.CONSOLIDATED_REPORT,
        (
            "CONSOLIDATEDFINANCIALSTATEMENTS",
            "CONSOLIDATEDANNUALREPORT",
            "КОНСОЛИДИРОВАННАЯФИНАНСОВАЯОТЧЁТНОСТЬ",
        ),
    ),
    (
        DocumentKind.FINANCIAL_NOTES,
        ("ПРИМЕЧАНИЯКФИНАНСОВОЙОТЧЁТНОСТИ", "NOTESTOTHEFINANCIALSTATEMENTS"),
    ),
    (
        DocumentKind.KYC_FILE,
        (
            "ДОСЬЕ«ЗНАЙСВОЕГОКЛИЕНТА»",
            "ПРОВЕРКАСВЯЗАННЫХСТОРОН",
            "KNOWYOURCUSTOMER",
            "RELATEDPARTYREVIEW",
        ),
    ),
    (
        DocumentKind.TREASURY_MEMO,
        ("СЛУЖЕБНАЯЗАПИСКАКАЗНАЧЕЙСТВА", "TREASURYMEMORANDUM"),
    ),
)

_SUPERSEDED_MARKERS = ("НЕДЕЙСТВУЮЩАЯРЕДАКЦИЯ", "НЕПРИМЕНЯЕТСЯ")
_DRAFT_MARKERS = ("НЕЯВЛЯЕТСЯОКОНЧАТЕЛЬНОЙПОЗИЦИЕЙ", "ПРОМЕЖУТОЧНАЯВЕДОМОСТЬ")
_FINAL_MARKERS = ("ОКОНЧАТЕЛЬНОЙПОЗИЦИЕЙАУДИТОРА",)


def squeeze(text: str) -> str:
    """Текст без пробелов и в верхнем регистре — форма для поиска маркеров."""
    return _WHITESPACE.sub("", text).upper()


def detect_kind(squeezed: str) -> DocumentKind:
    for kind, markers in _KIND_MARKERS:
        if any(marker in squeezed for marker in markers):
            return kind
    return DocumentKind.UNRELATED


def detect_status(squeezed: str, kind: DocumentKind) -> DocumentStatus:
    if any(marker in squeezed for marker in _SUPERSEDED_MARKERS):
        return DocumentStatus.SUPERSEDED
    if kind is DocumentKind.AUDIT_PROCEDURES:
        if any(marker in squeezed for marker in _DRAFT_MARKERS):
            return DocumentStatus.DRAFT
        if any(marker in squeezed for marker in _FINAL_MARKERS):
            return DocumentStatus.FINAL
    return DocumentStatus.CURRENT


def detect_account(text: str) -> str | None:
    """Счёт документа — наиболее часто упоминаемый.

    Единственного вхождения недостаточно: договор ссылается на счёт в шапке, в
    преамбуле и в колонтитуле, а посторонняя записка может упомянуть чужой счёт
    мимоходом.
    """
    found: list[str] = ACCOUNT_PATTERN.findall(text)
    if not found:
        return None
    ranked = Counter(found).most_common()
    best = max(count for _, count in ranked)
    tied = [account for account, count in ranked if count == best]
    return min(tied, key=text.index) if len(tied) > 1 else tied[0]


def detect_report_number(text: str) -> str | None:
    match = REPORT_PATTERN.search(text)
    return str(match.group(0)) if match else None


def detect_organisation(text: str) -> str | None:
    """Название организации, которой принадлежит документ.

    Берётся из пары «Организация — Счёт» в досье: она объявлена явно, в отличие от
    названий, разбросанных по тексту договора вперемешку с контрагентами.
    """
    match = _ORGANISATION.search(text)
    return match.group("name").strip() if match else None


def mentioned_organisations(text: str, known: Mapping[str, str]) -> frozenset[str]:
    """Счета тех организаций, чьи названия встретились в тексте.

    Нужно для документов, не содержащих номера счёта вовсе. Консолидированная
    отчётность материнской компании — как раз такой случай: с заёмщиком её связывает
    только упоминание его названия в примечании о сегментах.

    Сопоставление идёт по нормализованной форме, той же, что у связанных сторон:
    юридическая форма пишется по-разному в разных документах.
    """
    haystack = normalise_counterparty(text)
    return frozenset(account for name, account in known.items() if name and name in haystack)
