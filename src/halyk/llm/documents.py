"""Как документы заёмщика попадают в запрос. Одинаково у компилятора и у resolver.

Батч собирается по заёмщику, и это единственное место, где решается, что модель
увидит. Два свойства держатся именно здесь.

Изоляция: чужой документ не отфильтровывается, а роняет сборку батча. Молчаливый
фильтр превратил бы ошибку привязки документа к счёту в пропавшую величину, а
пропавшую величину видно только в конце, по несчитанной ячейке.

Детерминированность: порядок документов и страниц задан явно. Порядок обхода
файловой системы не воспроизводится между машинами, а он входит в ключ кэша — при
неявном порядке повторный прогон на тех же данных платил бы за те же ответы.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from halyk.models.document import DocumentFacts


class BorrowerIsolationError(RuntimeError):
    """В батч одного заёмщика попал документ другого."""


def own_documents(account_id: str, documents: Iterable[DocumentFacts]) -> tuple[DocumentFacts, ...]:
    """Документы заёмщика в стабильном порядке. Чужой документ — отказ, а не пропуск."""
    ordered = sorted(documents, key=lambda document: document.file_name)
    alien = [document.file_name for document in ordered if document.account_id != account_id]
    if alien:
        raise BorrowerIsolationError(
            f"в батч {account_id} попали документы других заёмщиков: {', '.join(alien)}"
        )
    return tuple(ordered)


def index(documents: Iterable[DocumentFacts]) -> dict[str, DocumentFacts]:
    """Документы по имени файла — так на них ссылается модель и так их ищет валидатор."""
    return {document.file_name: document for document in documents}


def render(documents: Sequence[DocumentFacts]) -> list[dict[str, Any]]:
    """Тело запроса: текст страниц вместе с адресом, по которому его потом проверят.

    Имя файла и номер страницы модель обязана вернуть обратно в source_refs, поэтому
    они лежат рядом с текстом. Хеша здесь нет намеренно: его мы подставим сами, а
    просить модель переписать шестьдесят четыре знака — верный способ получить
    правдоподобный чужой адрес.
    """
    return [
        {
            "file_name": document.file_name,
            "kind": document.kind.value,
            "status": document.status.value,
            "language": document.language_hint,
            "pages": [
                {"number": page.number, "text": page.text}
                for page in sorted(document.pages, key=lambda page: page.number)
            ],
        }
        for document in documents
    ]


def source_hashes(documents: Iterable[DocumentFacts]) -> tuple[str, ...]:
    """Отпечатки исходников для ключа кэша: другой датасет обязан дать промах."""
    return tuple(sorted(document.sha256 for document in documents))
