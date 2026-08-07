"""Решение о том, нужен ли странице OCR.

Правило смотрит на текст и на растры вместе. Одной длины текста мало: в этом
датасете есть и короткие честные страницы без картинок, которым OCR не нужен, и
страницы с заголовком в текстовом слое и полностраничным сканом под ним.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from halyk.models.document import OcrReason, PageFacts

# Порог подобран по открытому датасету: честные текстовые страницы там начинаются
# от нескольких сотен символов, а страницы-обложки скана дают заголовок и номер.
THIN_TEXT_CHARS = 200

# Растр меньше половины страницы — это логотип, печать или подпись. Текста в них
# нет, и гонять их через OCR значит платить за шум.
SUBSTANTIAL_IMAGE_AREA = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class OcrPlan:
    """Что именно отправлять в OCR: страницу целиком или встроенный растр."""

    reason: OcrReason
    xref: int | None = None

    @property
    def required(self) -> bool:
        return self.reason is not OcrReason.NONE


def plan_page(page: PageFacts) -> OcrPlan:
    substantial = [image for image in page.images if image.area_ratio >= SUBSTANTIAL_IMAGE_AREA]

    # Растр, не выведенный ни на одной странице, проверяем первым: рендер вернёт
    # пустой лист, и распознавать надо само изображение, а не то, что видно.
    if hidden := [image for image in substantial if not image.drawn]:
        return OcrPlan(reason=OcrReason.HIDDEN_IMAGE, xref=hidden[0].xref)

    if not substantial:
        return OcrPlan(reason=OcrReason.NONE)
    if page.char_count == 0:
        return OcrPlan(reason=OcrReason.NO_TEXT)
    if page.char_count < THIN_TEXT_CHARS:
        return OcrPlan(reason=OcrReason.THIN_TEXT_OVER_IMAGE)
    return OcrPlan(reason=OcrReason.NONE)
