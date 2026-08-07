"""Извлечение страниц PDF: текст, растры и их геометрия.

Растры описываются подробнее, чем нужно для простого случая, потому что решение
об OCR принимается именно по ним: страница с полностраничным сканом и страница с
логотипом внизу выглядят одинаково, если смотреть только на длину текста.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pymupdf

from halyk.models.document import PageFacts, PageImage

_CMYK_CHANNELS = 4


def _page_area(page: pymupdf.Page) -> Decimal:
    return Decimal(str(page.rect.width)) * Decimal(str(page.rect.height))


def _drawn_ratio(page: pymupdf.Page, xref: int) -> Decimal:
    area = _page_area(page)
    if area <= 0:
        return Decimal(0)
    covered = sum(
        Decimal(str(r.width)) * Decimal(str(r.height)) for r in page.get_image_rects(xref)
    )
    return min(Decimal(1), covered / area)


def _orphan_ratio(page: pymupdf.Page, width: int, height: int) -> Decimal:
    """Оценка площади для растра, не попавшего в поток вывода ни одной страницы.

    Считаем, что скан снят около 150 dpi, и переводим его пиксели в типографские
    точки: так полностраничный лист отличается от мелкой служебной картинки.
    """
    area = _page_area(page)
    if area <= 0 or not width or not height:
        return Decimal(0)
    at_150_dpi = Decimal(width) * Decimal(height) / Decimal(150 * 150) * Decimal(72 * 72)
    return min(Decimal(1), at_150_dpi / area)


def read_pages(path: Path) -> tuple[PageFacts, ...]:
    """Разбирает страницы, различая растры по месту их фактической отрисовки.

    `page.get_images()` перечисляет ресурсы страницы, а не то, что на ней нарисовано:
    в этом датасете один и тот же скан числится у всех страниц документа, хотя выведен
    ровно на одной. Принадлежность определяет `get_image_rects`, иначе каждая
    текстовая страница такого документа выглядела бы как скан.
    """
    with pymupdf.open(path) as document:
        drawn_anywhere = {
            xref
            for page in document
            for xref, *_ in page.get_images(full=True)
            if page.get_image_rects(xref)
        }

        pages, orphans_seen = [], set()
        for index, page in enumerate(document, start=1):
            text = page.get_text()
            images = []
            for xref, _, width, height, *_ in page.get_images(full=True):
                if ratio := _drawn_ratio(page, xref):
                    images.append(
                        PageImage(
                            xref=xref, width=width, height=height, drawn=True, area_ratio=ratio
                        )
                    )
                elif xref not in drawn_anywhere and xref not in orphans_seen:
                    # Растр, которого нет ни на одной странице. В открытом датасете
                    # такого не встретилось, но данные в нём были бы невидимы для
                    # рендера, поэтому относим его к первой странице-владельцу.
                    orphans_seen.add(xref)
                    images.append(
                        PageImage(
                            xref=xref,
                            width=width,
                            height=height,
                            drawn=False,
                            area_ratio=_orphan_ratio(page, width, height),
                        )
                    )
            pages.append(
                PageFacts(
                    number=index,
                    char_count=len(text.strip()),
                    images=tuple(images),
                    text=text,
                )
            )
    return tuple(pages)


def render_page_png(path: Path, number: int, dpi: int = 200) -> bytes:
    """Растр страницы целиком — вход для OCR, когда текста нет вовсе."""
    with pymupdf.open(path) as document:
        return bytes(document[number - 1].get_pixmap(dpi=dpi).tobytes("png"))


def extract_embedded_png(path: Path, xref: int) -> bytes:
    """Сам встроенный растр, минуя поток страницы.

    Нужен для скрытых изображений: рендер страницы вернул бы пустой лист, а данные
    лежат именно здесь.
    """
    with pymupdf.open(path) as document:
        pixmap = pymupdf.Pixmap(document, xref)
        # CMYK и другие многоканальные пространства PNG не принимает — переводим в RGB.
        if pixmap.colorspace is not None and pixmap.n - pixmap.alpha >= _CMYK_CHANNELS:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        return bytes(pixmap.tobytes("png"))
