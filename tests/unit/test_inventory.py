import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pymupdf
import pytest

from halyk.ingest.guard import dataset_files
from halyk.ingest.inventory import InventoryError, build_inventory, find_ledger, read_document
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.models.document import DocumentKind, DocumentStatus, PageFacts, TextSource
from halyk.parsing import pdf
from halyk.parsing.ocr import CachedOcr, OcrResponse


class StubOcrEngine:
    """Движок с заранее известным ответом и заранее известным расходом."""

    name = "stub-ocr"
    cache_signature: ClassVar[dict[str, object]] = {"variant": "stub"}

    def recognise(self, image: bytes) -> OcrResponse:
        return OcrResponse(
            text="Recognised page body.",
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            request_id="req_1",
        )


LEDGER = (
    "txn_id,date,account_id,counterparty,description,amount,currency\n"
    "TXN-P1-0001,2025-01-05,ACC-7801,Alpha LLP,Sales settlement,7000000.00,USD\n"
    "TXN-B1-0001,2025-02-05,ACC-7201,Beta LLP,Sales settlement,9000000.00,USD\n"
    "TXN-9001-0001,2025-03-05,ACC-9001,Noise Co,Rent,-100.00,USD\n"
)

# Текст синтетических PDF — латиница: встроенные шрифты PyMuPDF не кодируют
# кириллицу, а подсовывать в тест системный TrueType значит привязать его к машине.
# Классификация по русским маркерам проверяется отдельно, на подменённом парсере и
# на реальном датасете.
FILLER = "Document body for parsing checks. Account ACC-7801 mentioned once."


def make_pdf(path: Path, pages: list[str], image_page: int | None = None) -> None:
    document = pymupdf.open()
    for index, text in enumerate(pages):
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontname="helv", fontsize=9)
        if image_page == index:
            pixmap = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 1200, 1600))
            pixmap.clear_with(200)
            page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    document.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "documents").mkdir(parents=True)
    (root / "master_ledger.csv").write_text(LEDGER, encoding="utf-8")
    (root / "ground_truth.json").write_text('{"scenarios": {}}', encoding="utf-8")
    (root / "documents" / "aaa.csv").write_text("timestamp,srcip\n2025-01-01,10.0.0.1\n", "utf-8")
    make_pdf(root / "documents" / "b1.pdf", [FILLER])
    make_pdf(root / "documents" / "b2.pdf", [FILLER, FILLER])
    make_pdf(root / "documents" / "b3.pdf", [FILLER])
    make_pdf(root / "documents" / "b4.pdf", ["", ""], image_page=0)
    return root


def test_ledger_is_found_by_columns_not_by_name(dataset: Path) -> None:
    assert find_ledger(dataset_files(dataset)).name == "master_ledger.csv"


def test_no_ledger_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(InventoryError):
        find_ledger([tmp_path / "x.csv"])


def test_inventory_maps_scenarios_to_accounts(dataset: Path) -> None:
    found = build_inventory(dataset, ["P1", "B1"])
    assert found.scenarios.scenario_to_account == {"B1": "ACC-7201", "P1": "ACC-7801"}
    assert len(found.documents) == 4


def stub_pages(text: str) -> Callable[..., tuple[PageFacts, ...]]:
    def read(_path: Path) -> tuple[PageFacts, ...]:
        return (PageFacts(number=1, char_count=len(text), text=text),)

    return read


@pytest.mark.parametrize(
    ("text", "kind", "status"),
    [
        (
            "Д О Г О В О Р  Б А Н К О В С К О Г О  З А Й М А. ЗАЁМ № ACC-7801",
            DocumentKind.LOAN_AGREEMENT,
            DocumentStatus.CURRENT,
        ),
        (
            "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.). ДОГОВОР БАНКОВСКОГО ЗАЙМА. ЗАЁМ № ACC-7201",
            DocumentKind.LOAN_AGREEMENT,
            DocumentStatus.SUPERSEDED,
        ),
        (
            "Отчёт о выполнении согласованных процедур. Счёт ACC-7201. "
            "Выводы являются окончательной позицией аудитора. AR-2025-0634",
            DocumentKind.AUDIT_PROCEDURES,
            DocumentStatus.FINAL,
        ),
        ("Руководство по бренду и логотипу", DocumentKind.UNRELATED, DocumentStatus.CURRENT),
    ],
)
def test_document_is_classified_from_its_text(
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    kind: DocumentKind,
    status: DocumentStatus,
) -> None:
    monkeypatch.setattr(pdf, "read_pages", stub_pages(text))
    document = read_document(dataset / "documents" / "b1.pdf")
    assert (document.kind, document.status) == (kind, status)


def test_account_is_recorded_only_for_relevant_documents(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Посторонняя записка тоже может упомянуть чужой счёт — привязывать её незачем.
    monkeypatch.setattr(pdf, "read_pages", stub_pages("Служебная записка. Счёт ACC-7801."))
    assert read_document(dataset / "documents" / "b1.pdf").account_id is None


def test_ground_truth_never_enters_the_inventory(dataset: Path) -> None:
    found = build_inventory(dataset, ["P1", "B1"])
    assert "ground_truth.json" not in found.skipped_files
    assert all("ground_truth" not in doc.file_name for doc in found.documents)


def test_scan_page_is_planned_for_ocr(dataset: Path) -> None:
    found = build_inventory(dataset, ["P1", "B1"])
    planned = {(name, page.number) for name, page in found.ocr_pages}
    assert planned == {("b4.pdf", 1)}


def test_without_engine_text_is_left_native(dataset: Path) -> None:
    found = build_inventory(dataset, ["P1", "B1"])
    scan = next(doc for doc in found.documents if doc.file_name == "b4.pdf")
    assert scan.pages[0].text_source is TextSource.NATIVE
    assert scan.pages[0].needs_ocr


def test_renaming_and_reordering_do_not_change_the_result(dataset: Path, tmp_path: Path) -> None:
    """Порядок обхода файловой системы не воспроизводится между машинами.

    Ответ не должен зависеть ни от него, ни от того, как названы обезличенные файлы.
    """
    before = build_inventory(dataset, ["P1", "B1"]).report()

    shuffled = tmp_path / "shuffled"
    (shuffled / "documents").mkdir(parents=True)
    (shuffled / "zzz_ledger.csv").write_text(LEDGER, encoding="utf-8")
    for old, new in (("b1", "zz9"), ("b2", "aa0"), ("b3", "mm5"), ("b4", "cc3")):
        (shuffled / "documents" / f"{new}.pdf").write_bytes(
            (dataset / "documents" / f"{old}.pdf").read_bytes()
        )
    after = build_inventory(shuffled, ["P1", "B1"]).report()

    assert before["kinds"] == after["kinds"]
    assert before["scenarios"] == after["scenarios"]
    assert before["pages"] == after["pages"]
    assert [item["reason"] for item in before["ocr_pages"]] == [
        item["reason"] for item in after["ocr_pages"]
    ]


def test_report_is_json_serialisable(dataset: Path) -> None:
    report = build_inventory(dataset, ["P1", "B1"]).report()
    assert json.loads(json.dumps(report, ensure_ascii=False))["pdf"] == 4


def test_unreadable_document_is_pending_not_resolved(dataset: Path) -> None:
    """Полностью отсканированный документ нельзя считать посторонним.

    До распознавания он выглядит так же, как рекламный буклет, поэтому «ноль
    неразрешённых» не должно читаться как «всё разобрано».
    """
    found = build_inventory(dataset, ["P1", "B1"])
    assert found.unresolved_documents == ()
    assert found.unclassified_pending_ocr == ("b4.pdf",)
    assert "b4.pdf" in found.pending_ocr


def test_document_hash_is_recorded(dataset: Path) -> None:
    document = read_document(dataset / "documents" / "b1.pdf")
    assert len(document.sha256) == 64


def test_report_carries_the_cost_of_every_recognised_page(dataset: Path, tmp_path: Path) -> None:
    """Расход на распознавание должен быть виден в артефакте, а не только в консоли.

    Иначе холодный прогон не с чем сравнивать: время и токены живут в терминале,
    который к следующему запуску уже закрыт.
    """
    ocr = CachedOcr(
        engine=StubOcrEngine(),
        cache=ModelCache(directory=tmp_path / "ocr", policy=CachePolicy.READ_WRITE),
    )
    report = build_inventory(dataset, ["P1", "B1"], ocr).report()

    assert report["ocr_usage"]["live_calls"] == 1
    assert report["ocr_usage"]["total_tokens"] == 1500
    call = report["ocr_calls"][0]
    assert (call["document"], call["page"], call["cache_hit"]) == ("b4.pdf", 1, False)
    assert call["request_id"] == "req_1"


def test_report_without_ocr_reports_no_calls(dataset: Path) -> None:
    report = build_inventory(dataset, ["P1", "B1"]).report()
    assert report["ocr_calls"] == []
    assert report["ocr_usage"]["live_calls"] == 0
