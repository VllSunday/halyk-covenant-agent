"""Проверки на открытом датасете. Пропускаются, если его нет рядом с репозиторием.

Сам датасет в git не попадает, поэтому тесты обязаны оставаться зелёными без него —
но когда он на месте, они ловят расхождение наших контрактов с файлами организаторов.
"""

import collections
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from halyk.cli import app
from halyk.config import Settings
from halyk.dev.scoring import load_ground_truth, score_submission
from halyk.ingest.inventory import DatasetInventory, build_inventory
from halyk.ingest.normalise import CovenantPeriod, NormalisedLedger, normalise
from halyk.knowledge.facts import FactSet, build_facts
from halyk.knowledge.kyc import KycError, parse_related_party_policy
from halyk.llm.cache import CachePolicy, ModelCache
from halyk.models.adjustment import AdjustmentAction, AdjustmentStatus
from halyk.models.document import DocumentKind, DocumentStatus
from halyk.models.fact import FactKind
from halyk.models.submission import Submission
from halyk.output.template import SubmissionTemplate
from halyk.parsing.ocr import CachedOcr, OpenAIVisionOcr

DATASET = Path(__file__).resolve().parents[2] / "agentic-bank-public"
TEMPLATE = DATASET / "submission_template.json"
GROUND_TRUTH = DATASET / "ground_truth.json"
OCR_CACHE = Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "ocr"
PERIOD = CovenantPeriod(start=date(2025, 1, 1), end=date(2025, 12, 31))

pytestmark = pytest.mark.skipif(not DATASET.is_dir(), reason="открытый датасет не распакован")

# Кэш распознавания в git не попадает, а без него отсканированные документы читать
# нечем. Проверки, которым он нужен, пропускаются, а не падают на чужой машине.
needs_ocr = pytest.mark.skipif(
    not OCR_CACHE.is_dir() or not any(OCR_CACHE.glob("*.json")),
    reason="нет локального кэша OCR: выполните halyk inventory --ocr",
)


def _facts_command(out_dir: Path) -> list[str]:
    return [
        "facts",
        "--dataset",
        str(DATASET),
        "--template",
        str(TEMPLATE),
        "--ocr",
        "--out",
        str(out_dir),
    ]


@pytest.fixture(scope="module")
def inventory() -> DatasetInventory:
    return build_inventory(DATASET, SubmissionTemplate.load(TEMPLATE).scenarios)


@pytest.fixture(scope="module")
def recognised() -> DatasetInventory:
    """Перепись с текстом отсканированных страниц, взятым только из кэша.

    Настройки берутся те же, что у боевой команды: ключ кэша включает модель и
    параметры вызова, и с другими значениями попадания не будет.
    """
    engine = CachedOcr(
        engine=OpenAIVisionOcr(config=Settings.from_env().ocr),
        cache=ModelCache(directory=OCR_CACHE, policy=CachePolicy.REPLAY),
    )
    return build_inventory(DATASET, SubmissionTemplate.load(TEMPLATE).scenarios, engine)


@pytest.fixture(scope="module")
def facts(recognised: DatasetInventory) -> FactSet:
    return build_facts(recognised)


@pytest.fixture(scope="module")
def ledger(recognised: DatasetInventory, facts: FactSet) -> NormalisedLedger:
    return normalise(recognised.ledger, facts.adjustments, PERIOD)


def perfect_submission() -> Submission:
    """Ответ, собранный прямо из ключа, — верхняя граница возможного результата."""
    answers: dict[str, dict[str, dict[str, object]]] = {}
    for (scenario, covenant), cell in load_ground_truth(GROUND_TRUTH).items():
        answers.setdefault(scenario, {})[covenant] = {
            "status": cell.status,
            "actual": cell.actual,
            "evidence_txn_id": cell.evidence_txn_id,
        }
    return Submission.model_validate(
        {
            "team": "halyk",
            "contact_email": "team@example.com",
            "model": "claude-opus-5",
            "answers": answers,
        }
    )


def test_template_and_key_cover_the_same_cells() -> None:
    template = SubmissionTemplate.load(TEMPLATE)
    assert template.key_set() == set(load_ground_truth(GROUND_TRUTH))


def test_key_scores_full_marks_against_itself() -> None:
    report = score_submission(perfect_submission(), load_ground_truth(GROUND_TRUTH))
    assert report.total == report.max_total
    assert report.exact_cells == len(report.cells)


def test_perfect_answer_passes_template_check() -> None:
    assert SubmissionTemplate.load(TEMPLATE).check(perfect_submission()) == []


def test_key_statuses_use_the_expected_spelling() -> None:
    statuses = {cell.status for cell in load_ground_truth(GROUND_TRUTH).values()}
    assert statuses <= {"COMPLIANT", "BREACH"}


def test_evidence_is_absent_in_most_cells() -> None:
    # Из этого распределения следует приоритет: точность числа весит больше улики.
    key = load_ground_truth(GROUND_TRUTH)
    with_evidence = sum(1 for cell in key.values() if cell.evidence_txn_id)
    assert with_evidence < len(key) / 2


def test_actual_values_are_positive() -> None:
    assert all(cell.actual >= Decimal(0) for cell in load_ground_truth(GROUND_TRUTH).values())


def test_every_pdf_and_page_is_accounted_for(inventory: DatasetInventory) -> None:
    assert len(inventory.documents) == 200
    assert inventory.page_count == 843


def test_ledger_is_found_and_read(inventory: DatasetInventory) -> None:
    assert inventory.ledger_path.name.endswith(".csv")
    assert len(inventory.ledger) == 1473


def test_all_scenarios_map_to_distinct_accounts(inventory: DatasetInventory) -> None:
    mapping = inventory.scenarios.scenario_to_account
    assert len(mapping) == 12
    assert len(set(mapping.values())) == 12


def test_no_relevant_document_is_left_without_an_account(inventory: DatasetInventory) -> None:
    assert inventory.unresolved_documents == ()


def test_each_borrower_has_a_current_agreement_and_notes(inventory: DatasetInventory) -> None:
    for account in inventory.scenarios.scenario_to_account.values():
        documents = inventory.by_account(account)
        current = [
            doc
            for doc in documents
            if doc.kind is DocumentKind.LOAN_AGREEMENT and doc.status is DocumentStatus.CURRENT
        ]
        notes = [doc for doc in documents if doc.kind is DocumentKind.FINANCIAL_NOTES]
        assert len(current) == 1, account
        assert len(notes) == 1, account


def test_superseded_editions_are_separated(inventory: DatasetInventory) -> None:
    # Ровно по одной недействующей редакции 2024 года на заёмщика.
    by_status = collections.Counter(
        doc.status for doc in inventory.relevant if doc.kind is DocumentKind.LOAN_AGREEMENT
    )
    assert by_status[DocumentStatus.CURRENT] == 12
    assert by_status[DocumentStatus.SUPERSEDED] == 12


def test_current_edition_is_chosen_by_document_markers(inventory: DatasetInventory) -> None:
    """У P5 две редакции с разными порогами, и выбирать между ними надо по тексту.

    Порог 9.00x стоит в действующей редакции, 9.75x — в заменённой. Ошибка выбора
    даёт верное число при неверном вердикте, то есть теряет ячейку целиком. Эталон
    здесь намеренно не открывается: признак должен быть документальным, иначе на
    приватном наборе его нечем будет применить.
    """
    editions = [
        doc
        for doc in inventory.relevant
        if doc.kind is DocumentKind.LOAN_AGREEMENT and doc.account_id == "ACC-7805"
    ]
    assert len(editions) == 2

    by_status = {doc.status: "".join(page.text for page in doc.pages) for doc in editions}
    assert "9.00x" in by_status[DocumentStatus.CURRENT]
    assert "9.75x" in by_status[DocumentStatus.SUPERSEDED]
    assert "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ" in by_status[DocumentStatus.SUPERSEDED]
    assert "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ" not in by_status[DocumentStatus.CURRENT]


def test_only_one_audit_report_is_final(inventory: DatasetInventory) -> None:
    reports = [doc for doc in inventory.relevant if doc.kind is DocumentKind.AUDIT_PROCEDURES]
    assert len(reports) == 6
    assert sum(doc.status is DocumentStatus.FINAL for doc in reports) == 1


def test_report_numbers_repeat_across_borrowers(inventory: DatasetInventory) -> None:
    """Номер заключения не уникален глобально.

    Один и тот же `AR-2025-0634` стоит у финального отчёта одного заёмщика и у
    черновика другого, поэтому связывать их можно только парой «счёт + номер».
    """
    numbers = [
        doc.report_number
        for doc in inventory.relevant
        if doc.kind is DocumentKind.AUDIT_PROCEDURES and doc.report_number
    ]
    assert len(numbers) != len(set(numbers))


def test_ocr_is_needed_for_six_pages_in_four_documents(inventory: DatasetInventory) -> None:
    pages = inventory.ocr_pages
    assert len(pages) == 6
    assert len({name for name, _ in pages}) == 4


def test_related_party_thresholds_differ_between_borrowers(inventory: DatasetInventory) -> None:
    thresholds = {}
    for scenario, account in inventory.scenarios.scenario_to_account.items():
        dossiers = [
            doc for doc in inventory.by_account(account) if doc.kind is DocumentKind.KYC_FILE
        ]
        if not dossiers:
            continue
        try:
            thresholds[scenario] = parse_related_party_policy(dossiers[0].text).threshold
        except KycError:
            continue

    # Два досье целиком отсканированы, их пороги достанет только OCR.
    assert len(thresholds) == 10
    assert len(set(thresholds.values())) > 1
    assert min(thresholds.values()) == Decimal("0.20")


def test_every_readable_dossier_hides_a_holding_just_below_its_threshold(
    inventory: DatasetInventory,
) -> None:
    """В каждом досье есть доля чуть ниже порога — заготовленная ловушка.

    Досье, у которых таблица отсканирована, здесь пропускаются: без OCR читать в
    них нечего.
    """
    checked = 0
    for account in inventory.scenarios.scenario_to_account.values():
        dossiers = [
            doc for doc in inventory.by_account(account) if doc.kind is DocumentKind.KYC_FILE
        ]
        if not dossiers:
            continue
        try:
            policy = parse_related_party_policy(dossiers[0].text)
        except KycError:
            continue
        assert policy.near_miss, account
        checked += 1
    assert checked == 10


@needs_ocr
def test_every_borrower_has_a_kyc_profile(facts: FactSet) -> None:
    profiles = [f for f in facts.facts if f.kind is FactKind.RELATED_PARTY_POLICY]
    assert len(profiles) == 12
    assert len({f.account_id for f in profiles}) == 12


@needs_ocr
def test_external_facts_are_extracted_with_provenance(facts: FactSet) -> None:
    """Разовые статьи, покрытие и обязательство — входы формулы, а не проводки."""
    kinds = collections.Counter(f.kind for f in facts.facts)
    assert kinds[FactKind.ONE_OFF_ITEM] == 3
    assert kinds[FactKind.ONE_OFF_POLICY] == 1
    assert kinds[FactKind.COLLATERAL_COVERAGE] == 1
    assert kinds[FactKind.AGGREGATE_OBLIGATION] == 1
    assert kinds[FactKind.FX_SETTLEMENT] == 1
    assert all(f.source.file_hash and f.source.page >= 1 for f in facts.facts)


@needs_ocr
def test_collateral_coverage_of_p9(facts: FactSet) -> None:
    fact = next(f for f in facts.facts if f.kind is FactKind.COLLATERAL_COVERAGE)
    assert fact.threshold == Decimal("0.50")
    assert [(s.counterparty, s.share) for s in fact.subsidiaries] == [
        ("Zhezkazgan Conveyor Assets LLP", Decimal("0.876")),
        ("Zhezkazgan Processing Holdings LLP", Decimal("0.114")),
    ]
    assert fact.unrestricted == ("Zhezkazgan Processing Holdings LLP",)


@needs_ocr
def test_one_off_items_of_p4(facts: FactSet) -> None:
    items = [f for f in facts.facts if f.kind is FactKind.ONE_OFF_ITEM]
    assert [str(f.amount.to_decimal()) for f in items] == [
        "251338.94",
        "342905.28",
        "481247.63",
    ]
    policy = next(f for f in facts.facts if f.kind is FactKind.ONE_OFF_POLICY)
    assert str(policy.minimum.to_decimal()) == "300000"


@needs_ocr
def test_only_the_final_report_changes_the_ledger(ledger: NormalisedLedger) -> None:
    """Пять ведомостей зафиксированы, применён один окончательный отчёт."""
    from_audit = [a for a in ledger.adjustments if a.source.kind is DocumentKind.AUDIT_PROCEDURES]
    assert len(from_audit) == 6
    applied = [a for a in from_audit if a.status is AdjustmentStatus.APPLIED]
    assert [a.txn_id for a in applied] == ["TXN-B1-0020"]
    assert applied[0].new_value == "Процентные расходы"
    assert applied[0].source.status is DocumentStatus.FINAL

    drafts = [a for a in from_audit if a.source.status is DocumentStatus.DRAFT]
    assert len(drafts) == 5
    assert collections.Counter(a.status for a in drafts) == {
        AdjustmentStatus.UNCONFIRMED: 4,
        AdjustmentStatus.SUPERSEDED: 1,
    }


@needs_ocr
def test_unconfirmed_drafts_leave_their_transactions_untouched(
    ledger: NormalisedLedger,
) -> None:
    for txn_id in ("TXN-P3-0001", "TXN-P6-0044", "TXN-P8-0016", "TXN-P9-0025", "TXN-B1-0023"):
        transaction = ledger.by_id(txn_id)
        assert transaction is not None, txn_id
        assert transaction.covenant_category is None, txn_id
        assert not transaction.is_adjusted, txn_id


@needs_ocr
def test_missing_amounts_are_restored_from_documents(ledger: NormalisedLedger) -> None:
    for txn_id, amount in (("TXN-P7-0033", "-486204.19"), ("TXN-P8-0031", "-884204.16")):
        transaction = ledger.by_id(txn_id)
        assert transaction is not None
        assert transaction.row.amount is None
        assert str(transaction.amount.to_decimal()) == amount  # type: ignore[union-attr]


@needs_ocr
def test_cutoff_moves_a_transaction_out_of_the_period(ledger: NormalisedLedger) -> None:
    deferred = ledger.by_id("TXN-P1-0045")
    excluded = ledger.by_id("TXN-B4-0026")
    assert deferred is not None and excluded is not None
    assert not deferred.in_period
    assert not excluded.in_period


@needs_ocr
def test_disclosed_rate_converts_the_only_euro_settlement(ledger: NormalisedLedger) -> None:
    converted = ledger.by_id("TXN-P3-0024")
    assert converted is not None
    assert str(converted.amount.to_decimal()) == "-710945.73"  # type: ignore[union-attr]
    assert converted.row.amount.currency.value == "EUR"  # type: ignore[union-attr]


@needs_ocr
def test_rejected_reviews_change_nothing(ledger: NormalisedLedger) -> None:
    rejected = [a for a in ledger.adjustments if a.status is AdjustmentStatus.REJECTED]
    assert {a.selector.txn_id for a in rejected} == {"TXN-P10-0012", "TXN-P10-0021"}
    for adjustment in rejected:
        assert ledger.by_id(adjustment.selector.txn_id or "").covenant_category is None  # type: ignore[union-attr]


@needs_ocr
def test_every_change_is_backed_by_a_document(ledger: NormalisedLedger) -> None:
    changed = [t for t in ledger.transactions if t.is_adjusted]
    applied = {a.id: a for a in ledger.applied}
    assert len(changed) == len(applied) == 8
    for transaction in changed:
        for key in transaction.adjustments:
            assert applied[key].source.file_hash
            assert applied[key].reason


@needs_ocr
def test_normalised_ledger_is_reproducible(recognised: DatasetInventory, facts: FactSet) -> None:
    first = normalise(recognised.ledger, facts.adjustments, PERIOD)
    second = normalise(recognised.ledger, facts.adjustments, PERIOD)
    assert [t.record() for t in first.transactions] == [t.record() for t in second.transactions]
    assert len(first.transactions) == len(recognised.ledger)


@needs_ocr
def test_nothing_is_left_unaccounted_for(ledger: NormalisedLedger, facts: FactSet) -> None:
    assert ledger.problems == ()
    assert facts.unparsed_dossiers == ()


@needs_ocr
def test_strict_run_passes_on_the_public_dataset(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, [*_facts_command(tmp_path), "--strict"])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["period"] == {"start": "2025-01-01", "end": "2025-12-31"}
    assert summary["problems"] == []
    assert summary["applied"] == 8


@needs_ocr
def test_strict_run_fails_when_a_document_stops_having_effect(tmp_path: Path) -> None:
    """Сузим период так, что ноябрьский аванс и без документа вне его.

    Исключать нечего, вывод примечаний ни на что не влияет — и строгий прогон
    обязан это заметить, а не отчитаться об успехе.
    """
    result = CliRunner().invoke(
        app, [*_facts_command(tmp_path), "--strict", "--period-end", "2025-11-19"]
    )
    assert result.exit_code == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in summary["problems"]] == ["ineffective"]


@needs_ocr
def test_actions_used_on_the_public_dataset(ledger: NormalisedLedger) -> None:
    used = collections.Counter(a.action for a in ledger.applied)
    assert used == {
        AdjustmentAction.RECLASSIFY: 3,
        AdjustmentAction.SET_MISSING_AMOUNT: 2,
        AdjustmentAction.EXCLUDE: 1,
        AdjustmentAction.SET_EFFECTIVE_DATE: 1,
        AdjustmentAction.CONVERT_CURRENCY: 1,
    }
