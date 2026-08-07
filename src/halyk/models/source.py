from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from halyk.models.document import DocumentKind, DocumentStatus


class BoundingBox(BaseModel):
    """Координаты в системе PyMuPDF: начало в левом верхнем углу, единица — пункт."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


class SourceAuthority(StrEnum):
    """Можно ли опираться на документ при расчёте.

    Промежуточная ведомость аудитора и заменённая редакция договора читаются и
    хранятся, но менять по ним ответ нельзя. Разделение статуса документа и права
    на него опираться нужно потому, что второе зависит не только от самого
    документа: черновик перестаёт действовать, когда у того же заёмщика появился
    окончательный отчёт.
    """

    AUTHORITATIVE = "authoritative"
    RECORDED = "recorded"
    IGNORED = "ignored"


class SourceRef(BaseModel):
    """Адрес фрагмента в исходном документе.

    Каждое утверждение системы адресуется этой тройкой, и по ней же строится вырезка
    страницы для карточки решения. Тип, статус и номер заключения носятся вместе с
    адресом: без них нельзя ответить, почему один вывод применён, а другой отклонён.
    """

    model_config = ConfigDict(frozen=True)

    file_hash: str = Field(min_length=64, max_length=64)
    file_name: str
    page: int = Field(ge=1)
    kind: DocumentKind = DocumentKind.UNRELATED
    status: DocumentStatus = DocumentStatus.CURRENT
    authority: SourceAuthority = SourceAuthority.RECORDED
    account_id: str | None = None
    report_number: str | None = None
    bbox: BoundingBox | None = None
    quote: str | None = None
