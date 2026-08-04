from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Координаты в системе PyMuPDF: начало в левом верхнем углу, единица — пункт."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


class SourceRef(BaseModel):
    """Адрес фрагмента в исходном документе.

    Каждое утверждение системы адресуется этой тройкой, и по ней же строится вырезка
    страницы для карточки решения.
    """

    model_config = ConfigDict(frozen=True)

    file_hash: str = Field(min_length=64, max_length=64)
    file_name: str
    page: int = Field(ge=1)
    bbox: BoundingBox | None = None
    quote: str | None = None
