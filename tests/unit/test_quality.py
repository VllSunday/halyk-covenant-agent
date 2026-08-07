from decimal import Decimal

from halyk.models.document import OcrReason, PageFacts, PageImage
from halyk.parsing.quality import plan_page


def image(area: str, drawn: bool = True, xref: int = 1) -> PageImage:
    return PageImage(xref=xref, width=1200, height=1600, drawn=drawn, area_ratio=Decimal(area))


def page(chars: int, *images: PageImage) -> PageFacts:
    return PageFacts(number=1, char_count=chars, images=tuple(images))


def test_plain_text_page_needs_nothing() -> None:
    assert plan_page(page(2500)).reason is OcrReason.NONE


def test_short_text_page_without_images_needs_nothing() -> None:
    """Короткая страница без картинок — это просто короткая страница.

    Раньше правило смотрело только на длину текста и звало OCR на служебные листы,
    в которых распознавать нечего.
    """
    assert plan_page(page(37)).reason is OcrReason.NONE


def test_logo_sized_image_is_not_a_scan() -> None:
    assert plan_page(page(2500, image("0.04"))).reason is OcrReason.NONE


def test_full_page_scan_without_text() -> None:
    assert plan_page(page(0, image("1.0"))).reason is OcrReason.NO_TEXT


def test_scan_under_a_heading() -> None:
    # Заголовок раздела в текстовом слое, содержимое — картинкой под ним.
    assert plan_page(page(37, image("0.65"))).reason is OcrReason.THIN_TEXT_OVER_IMAGE


def test_large_image_beside_full_text_is_left_alone() -> None:
    assert plan_page(page(2500, image("0.65"))).reason is OcrReason.NONE


def test_undrawn_image_is_recognised_directly() -> None:
    plan = plan_page(page(0, image("0.9", drawn=False, xref=42)))
    assert plan.reason is OcrReason.HIDDEN_IMAGE
    assert plan.xref == 42


def test_drawn_scan_is_rendered_not_extracted() -> None:
    assert plan_page(page(0, image("1.0"))).xref is None
