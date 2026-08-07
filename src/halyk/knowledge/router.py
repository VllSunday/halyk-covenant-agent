"""Определение типа, принадлежности и статуса документа по его содержимому.

Имена файлов обезличены, поэтому единственный источник — текст. Правила здесь
наблюдённые, а не взятые из условия задачи, поэтому каждое из них может не
сработать на другом наборе: нераспознанное не подменяется догадкой, а попадает в
отчёт отдельным списком.
"""

from __future__ import annotations

import re
from collections import Counter

from halyk.models.document import DocumentKind, DocumentStatus

ACCOUNT_PATTERN = re.compile(r"ACC-\d{4}")
REPORT_PATTERN = re.compile(r"\bAR-\d{4}-\d{4}\b")
_WHITESPACE = re.compile(r"\s+")

# Маркеры сравниваются с текстом, из которого убраны все пробелы. В извлечённом
# слое заголовки идут вразрядку («Д О Г О В О Р») и рвутся посреди слова
# («КОНФИДЕНЦИА ЛЬНО»), и обычный поиск подстроки на них не работает.
_KIND_MARKERS: tuple[tuple[DocumentKind, tuple[str, ...]], ...] = (
    (DocumentKind.LOAN_AGREEMENT, ("ДОГОВОРБАНКОВСКОГОЗАЙМА",)),
    (
        DocumentKind.AUDIT_PROCEDURES,
        ("ОТЧЁТОВЫПОЛНЕНИИСОГЛАСОВАННЫХПРОЦЕДУР", "ВЕДОМОСТЬВОПРОСОВПОКЛАССИФИКАЦИИ"),
    ),
    (DocumentKind.FINANCIAL_NOTES, ("ПРИМЕЧАНИЯКФИНАНСОВОЙОТЧЁТНОСТИ",)),
    (DocumentKind.KYC_FILE, ("ДОСЬЕ«ЗНАЙСВОЕГОКЛИЕНТА»", "ПРОВЕРКАСВЯЗАННЫХСТОРОН")),
    (DocumentKind.TREASURY_MEMO, ("СЛУЖЕБНАЯЗАПИСКАКАЗНАЧЕЙСТВА",)),
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
