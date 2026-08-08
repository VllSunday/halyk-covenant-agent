"""Что мы знаем о документе после разбора, до всякой интерпретации."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocumentKind(StrEnum):
    """Тип документа. Определяется по содержимому: имена файлов обезличены."""

    LOAN_AGREEMENT = "loan_agreement"
    AUDIT_PROCEDURES = "audit_procedures"
    FINANCIAL_NOTES = "financial_notes"
    KYC_FILE = "kyc_file"
    TREASURY_MEMO = "treasury_memo"
    # Отчётность материнской компании: относится к заёмщику, но номера его счёта не
    # содержит и привязывается по названию организации.
    CONSOLIDATED_REPORT = "consolidated_report"
    UNRELATED = "unrelated"

    @property
    def is_relevant(self) -> bool:
        return self is not DocumentKind.UNRELATED


class DocumentStatus(StrEnum):
    """Действует ли документ и является ли он окончательным.

    Две оси уложены в одну шкалу намеренно: у договора значимо, не заменён ли он
    новой редакцией, у аудиторского отчёта — не является ли он черновиком. Обе
    отвечают на один вопрос — можно ли на документ опираться.
    """

    CURRENT = "current"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    FINAL = "final"

    @property
    def is_authoritative(self) -> bool:
        return self in (DocumentStatus.CURRENT, DocumentStatus.FINAL)


class TextSource(StrEnum):
    """Откуда взялся текст страницы. Пишется в lineage вместе с самим текстом."""

    NATIVE = "native"
    RENDERED_OCR = "rendered_ocr"
    EMBEDDED_IMAGE_OCR = "embedded_image_ocr"


class OcrReason(StrEnum):
    """Почему страницу отправили в OCR. Пустая причина означает, что не отправляли."""

    NONE = "none"
    NO_TEXT = "no_text"
    THIN_TEXT_OVER_IMAGE = "thin_text_over_image"
    HIDDEN_IMAGE = "hidden_image"


class PageImage(BaseModel):
    """Растр, встроенный в страницу.

    `drawn` различает картинку, которую видно при рендере, и ту, что лежит в
    ресурсах, но в поток страницы не попала: в примечаниях одного из заёмщиков
    именно так спрятан полностраничный скан с числами.
    """

    model_config = ConfigDict(frozen=True)

    xref: int
    width: int
    height: int
    drawn: bool
    area_ratio: Decimal = Field(ge=0)


class PageFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    char_count: int = Field(ge=0)
    images: tuple[PageImage, ...] = ()
    text_source: TextSource = TextSource.NATIVE
    ocr_reason: OcrReason = OcrReason.NONE
    text: str = ""

    @property
    def needs_ocr(self) -> bool:
        return self.ocr_reason is not OcrReason.NONE

    @property
    def largest_image_area(self) -> Decimal:
        return max((image.area_ratio for image in self.images), default=Decimal(0))


class DocumentFacts(BaseModel):
    """Разобранный документ вместе с тем, как именно он был разобран."""

    model_config = ConfigDict(frozen=True)

    file_name: str
    sha256: str
    kind: DocumentKind
    status: DocumentStatus
    account_id: str | None = None
    report_number: str | None = None
    pages: tuple[PageFacts, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def letters(self) -> tuple[int, int]:
        """Латиница и кириллица по буквам.

        Считаются только буквы: цифры, суммы и пунктуация одинаковы в обоих языках и
        размыли бы долю до бессмысленной.
        """
        latin = cyrillic = 0
        for char in self.text:
            if "a" <= (lowered := char.lower()) <= "z":
                latin += 1
            elif "а" <= lowered <= "я" or lowered == "ё":
                cyrillic += 1
        return latin, cyrillic

    @property
    def language_hint(self) -> str:
        """Подсказка о языке документа, а не проверка.

        Нужна затем, чтобы англоязычный документ было видно в отчёте сразу, а не по
        пропавшей ячейке: ровно так мы потеряли консолидированную отчётность. Именно
        подсказка — двуязычный документ или английское название внутри русского текста
        нормальны и прогон ронять не должны.
        """
        latin, cyrillic = self.letters
        if not latin and not cyrillic:
            return "unknown"
        if not cyrillic:
            return "en"
        if not latin:
            return "ru"
        return "mixed" if latin > cyrillic / 4 else "ru"

    @property
    def pages_needing_ocr(self) -> tuple[PageFacts, ...]:
        return tuple(page for page in self.pages if page.needs_ocr)
