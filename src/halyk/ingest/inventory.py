"""Перепись датасета: что за файлы пришли, к кому относятся и как разобраны.

Результат этого шага — единственный вход для всего, что считается дальше. Поэтому
он детерминирован по построению, а нераспознанное попадает в отдельные списки, а не
растворяется в общем итоге.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halyk.hashing import sha256_file
from halyk.ingest.guard import dataset_files
from halyk.ingest.ledger import LedgerError, read_ledger
from halyk.ingest.scenarios import ScenarioMap
from halyk.knowledge import router
from halyk.knowledge.kyc import normalise_counterparty
from halyk.models.document import DocumentFacts, PageFacts, TextSource
from halyk.models.transaction import LedgerRow
from halyk.parsing import pdf
from halyk.parsing.ocr import CachedOcr, OcrCall, summarise_calls
from halyk.parsing.quality import OcrPlan, plan_page


class InventoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    documents: tuple[DocumentFacts, ...]
    ledger: tuple[LedgerRow, ...]
    scenarios: ScenarioMap
    ledger_path: Path
    skipped_files: tuple[str, ...] = ()
    unresolved_documents: tuple[str, ...] = field(default=())
    ocr_calls: tuple[OcrCall, ...] = field(default=())

    @property
    def relevant(self) -> tuple[DocumentFacts, ...]:
        return tuple(doc for doc in self.documents if doc.kind.is_relevant)

    def by_account(self, account_id: str) -> tuple[DocumentFacts, ...]:
        return tuple(doc for doc in self.relevant if doc.account_id == account_id)

    @property
    def page_count(self) -> int:
        return sum(len(doc.pages) for doc in self.documents)

    @property
    def ocr_pages(self) -> tuple[tuple[str, PageFacts], ...]:
        return tuple(
            (doc.file_name, page) for doc in self.documents for page in doc.pages_needing_ocr
        )

    @property
    def pending_ocr(self) -> tuple[str, ...]:
        return tuple(doc.file_name for doc in self.documents if doc.pages_needing_ocr)

    @property
    def unclassified_pending_ocr(self) -> tuple[str, ...]:
        """Документы, тип которых не определился из-за отсутствия текста.

        Считать их посторонними нельзя: полностью отсканированное досье выглядит
        ровно так же, как рекламный буклет, и без распознавания отличить их нечем.
        Отдельная метрика нужна, чтобы «ноль неразрешённых» не читалось как
        «всё разобрано».
        """
        return tuple(
            doc.file_name
            for doc in self.documents
            if doc.pages_needing_ocr and not doc.kind.is_relevant
        )

    def report(self) -> dict[str, Any]:
        """Сводка для артефакта прогона. Тексты страниц сюда не попадают."""
        kinds = Counter(doc.kind.value for doc in self.documents)
        return {
            "files": len(self.documents) + len(self.skipped_files),
            "pdf": len(self.documents),
            "pages": self.page_count,
            "kinds": dict(sorted(kinds.items())),
            "scenarios": dict(sorted(self.scenarios.scenario_to_account.items())),
            "ocr_pages": [
                {"document": name, "page": page.number, "reason": page.ocr_reason.value}
                for name, page in self.ocr_pages
            ],
            "unresolved_documents": list(self.unresolved_documents),
            "unclassified_pending_ocr": list(self.unclassified_pending_ocr),
            # Расход и время распознавания попадают в артефакт, а не только в
            # терминал: сравнивать холодный прогон с повторным потом будет нечем.
            "ocr_calls": [call.report() for call in self.ocr_calls],
            "ocr_usage": summarise_calls(self.ocr_calls),
        }


def find_ledger(paths: Sequence[Path]) -> Path:
    """Реестр опознаётся по набору колонок, а не по имени файла.

    Имена в датасете обезличены, а посторонних CSV в нём хватает — например,
    выгрузка логов веб-сервера.
    """
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        try:
            read_ledger(path)
        except (LedgerError, UnicodeDecodeError, KeyError):
            continue
        return path
    raise InventoryError("Среди файлов датасета нет реестра транзакций")


def _apply_ocr(source: Path, page: PageFacts, plan: OcrPlan, ocr: CachedOcr) -> PageFacts:
    if plan.xref is not None:
        image = pdf.extract_embedded_png(source, plan.xref)
        origin = TextSource.EMBEDDED_IMAGE_OCR
    else:
        image = pdf.render_page_png(source, page.number)
        origin = TextSource.RENDERED_OCR

    recognised = ocr.recognise(image, document=source.name, page=page.number)
    # Распознанный текст добавляется к найденному, а не затирает его: на странице со
    # скрытым сканом в текстовом слое остаётся заголовок раздела, и он тоже нужен.
    merged = f"{page.text}\n{recognised}".strip() if page.text.strip() else recognised
    return page.model_copy(
        update={"text": merged, "char_count": len(merged.strip()), "text_source": origin}
    )


def read_document(path: Path, ocr: CachedOcr | None = None) -> DocumentFacts:
    pages = []
    for parsed in pdf.read_pages(path):
        plan = plan_page(parsed)
        page = parsed.model_copy(update={"ocr_reason": plan.reason})
        if plan.required and ocr is not None:
            page = _apply_ocr(path, page, plan, ocr)
        pages.append(page)

    text = "\n".join(page.text for page in pages)
    squeezed = router.squeeze(text)
    kind = router.detect_kind(squeezed)
    return DocumentFacts(
        file_name=path.name,
        sha256=sha256_file(path),
        kind=kind,
        status=router.detect_status(squeezed, kind),
        account_id=router.detect_account(text) if kind.is_relevant else None,
        report_number=router.detect_report_number(text),
        pages=tuple(pages),
    )


def _link_by_organisation(documents: Sequence[DocumentFacts]) -> list[DocumentFacts]:
    """Второй проход: привязать документы, в которых нет номера счёта.

    Отчётность материнской компании относится к заёмщику, но его счёта не называет —
    связь идёт через упоминание названия организации в примечании о сегментах. Пока
    привязка шла только по `ACC-NNNN`, такой документ уходил в шум молча, а вместе с
    ним и величина, без которой не считается ковенант.

    Привязка принимается только при единственном совпадении: документ, упомянувший
    двух заёмщиков сразу, остаётся неразрешённым и попадает в отчёт. Угадывать здесь
    хуже, чем показать человеку.
    """
    known = {
        normalise_counterparty(name): doc.account_id
        for doc in documents
        if doc.account_id and (name := router.detect_organisation(doc.text))
    }
    if not known:
        return list(documents)

    linked = []
    for document in documents:
        if document.account_id or not document.kind.is_relevant:
            linked.append(document)
            continue
        accounts = router.mentioned_organisations(document.text, known)
        linked.append(
            document.model_copy(update={"account_id": next(iter(accounts))})
            if len(accounts) == 1
            else document
        )
    return linked


def build_inventory(
    root: Path, template_scenarios: Sequence[str], ocr: CachedOcr | None = None
) -> DatasetInventory:
    files = dataset_files(root)
    ledger_path = find_ledger(files)
    ledger = read_ledger(ledger_path)

    documents = [read_document(path, ocr) for path in files if path.suffix.lower() == ".pdf"]
    skipped = tuple(p.name for p in files if p.suffix.lower() != ".pdf")
    documents = _link_by_organisation(documents)

    unresolved = tuple(
        doc.file_name for doc in documents if doc.kind.is_relevant and doc.account_id is None
    )
    return DatasetInventory(
        documents=tuple(documents),
        ledger=ledger,
        scenarios=ScenarioMap.build(template_scenarios, ledger),
        ledger_path=ledger_path,
        skipped_files=skipped,
        unresolved_documents=unresolved,
        ocr_calls=tuple(ocr.records) if ocr is not None else (),
    )
