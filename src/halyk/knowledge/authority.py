"""Кто из документов имеет право менять ответ.

Правило одно: опираемся на действующую редакцию и на окончательный вывод. Всё
остальное читается и хранится, но расчёт не меняет. Тонкость в том, что номер
заключения не уникален: один и тот же `AR-2025-0634` стоит у окончательного отчёта
одного заёмщика и у промежуточной ведомости другого. Поэтому отчёт ищется парой
«счёт + номер», а не номером.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus
from halyk.models.source import SourceAuthority

# Примечания ссылаются на отчёт, которым следует руководствоваться: «изложен в отчёте
# о выполнении согласованных процедур № AR-2025-0634».
_REPORT_REFERENCE = re.compile(r"согласованных процедур\s*№\s*(?P<number>[A-Z]{2}-\d{4}-\d{4})")


def resolve_authority(document: DocumentFacts) -> SourceAuthority:
    """Право документа менять расчёт, исходя только из него самого."""
    if not document.kind.is_relevant:
        return SourceAuthority.IGNORED
    if document.status is DocumentStatus.SUPERSEDED:
        return SourceAuthority.IGNORED
    if document.status is DocumentStatus.DRAFT:
        return SourceAuthority.RECORDED
    return SourceAuthority.AUTHORITATIVE


def final_reports(documents: Iterable[DocumentFacts]) -> dict[tuple[str, str], DocumentFacts]:
    """Окончательные отчёты по паре «счёт + номер»."""
    return {
        (doc.account_id, doc.report_number): doc
        for doc in documents
        if doc.kind is DocumentKind.AUDIT_PROCEDURES
        and doc.status is DocumentStatus.FINAL
        and doc.account_id
        and doc.report_number
    }


def referenced_report(text: str) -> str | None:
    """Номер отчёта, на который ссылаются примечания к отчётности."""
    match = _REPORT_REFERENCE.search(text)
    return match.group("number") if match else None


def authoritative_report(
    account_id: str, text: str, documents: Sequence[DocumentFacts]
) -> DocumentFacts | None:
    """Отчёт, которым велят руководствоваться примечания этого заёмщика.

    Проверка идёт по обеим частям пары. Совпадение одного номера означало бы, что
    вывод по чужому заёмщику применяется к нашему, а именно на этом датасет и ловит.
    """
    number = referenced_report(text)
    if number is None:
        return None
    return final_reports(documents).get((account_id, number))


def superseded_drafts(documents: Sequence[DocumentFacts]) -> frozenset[str]:
    """Ведомости, у чьего заёмщика есть окончательный отчёт.

    Такая ведомость заменена целиком: её предварительная позиция не переносится, а
    первоначальная классификация сохраняется, если окончательный отчёт её не менял.
    """
    with_final = {
        doc.account_id
        for doc in documents
        if doc.kind is DocumentKind.AUDIT_PROCEDURES and doc.status is DocumentStatus.FINAL
    }
    return frozenset(
        doc.file_name
        for doc in documents
        if doc.kind is DocumentKind.AUDIT_PROCEDURES
        and doc.status is DocumentStatus.DRAFT
        and doc.account_id in with_final
    )
