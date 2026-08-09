"""Матрица семантической эквивалентности: русский и английский дают одну модель.

Организатор подтвердил, что документы приватного набора могут быть на английском.
Отдельной английской модели нет намеренно — каждая пара фикстур обязана дать
идентичный `Fact` или `Adjustment`, иначе две ветви разойдутся на первой же правке.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from halyk.knowledge import notes
from halyk.knowledge.authority import referenced_report
from halyk.knowledge.classifier import _STATED_CATEGORIES
from halyk.knowledge.kyc import parse_collateral_policy, parse_related_party_policy
from halyk.knowledge.ppe import parse_ppe_movement
from halyk.knowledge.router import detect_kind, detect_organisation, detect_status, squeeze
from halyk.models.document import DocumentKind, DocumentStatus
from halyk.money import Currency, Money


def item(body: str) -> notes.Disclosure:
    return notes.Disclosure(number="9.1", body=notes.flatten(body))


# --- тип, статус и принадлежность документа ----------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Д О Г О В О Р   Б А Н К О В С К О Г О   З А Й М А", DocumentKind.LOAN_AGREEMENT),
        ("BANK LOAN AGREEMENT", DocumentKind.LOAN_AGREEMENT),
        ("ПРИМЕЧАНИЯ К ФИНАНСОВОЙ ОТЧЁТНОСТИ", DocumentKind.FINANCIAL_NOTES),
        ("NOTES TO THE FINANCIAL STATEMENTS", DocumentKind.FINANCIAL_NOTES),
        ("ДОСЬЕ «ЗНАЙ СВОЕГО КЛИЕНТА»", DocumentKind.KYC_FILE),
        ("KNOW YOUR CUSTOMER FILE", DocumentKind.KYC_FILE),
        ("ОТЧЁТ О ВЫПОЛНЕНИИ СОГЛАСОВАННЫХ ПРОЦЕДУР", DocumentKind.AUDIT_PROCEDURES),
        ("AGREED-UPON PROCEDURES REPORT", DocumentKind.AUDIT_PROCEDURES),
        ("СЛУЖЕБНАЯ ЗАПИСКА КАЗНАЧЕЙСТВА", DocumentKind.TREASURY_MEMO),
        ("TREASURY MEMORANDUM", DocumentKind.TREASURY_MEMO),
        ("CONSOLIDATED FINANCIAL STATEMENTS", DocumentKind.CONSOLIDATED_REPORT),
    ],
)
def test_document_kind_is_language_neutral(text: str, kind: DocumentKind) -> None:
    assert detect_kind(squeeze(text)) is kind


@pytest.mark.parametrize(
    "text",
    ["НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.)", "SUPERSEDED EDITION (2024)", "NOT OPERATIVE"],
)
def test_superseded_edition_is_language_neutral(text: str) -> None:
    marked = squeeze(f"ДОГОВОР БАНКОВСКОГО ЗАЙМА {text}")
    assert detect_status(marked, DocumentKind.LOAN_AGREEMENT) is DocumentStatus.SUPERSEDED


@pytest.mark.parametrize(
    "clause",
    [
        "пока коэффициент долговой нагрузки не превышает 3.00x, указанное "
        "ограничение капитальных затрат не применяется",
        "while the leverage ratio does not exceed 3.00x, the stated capital "
        "expenditure limitation does not apply",
    ],
)
def test_conditional_carve_out_does_not_supersede_agreement(clause: str) -> None:
    """«Не применяется» относится к ковенанту, а не к редакции договора.

    Приняв это за маркер вытеснения, разбор оставляет заёмщика вовсе без
    действующего договора — и ошибка тихая: договор в наборе есть.
    """
    marked = squeeze(f"ДОГОВОР БАНКОВСКОГО ЗАЙМА Пункт 6.1. {clause}.")
    assert detect_status(marked, DocumentKind.LOAN_AGREEMENT) is DocumentStatus.CURRENT


@pytest.mark.parametrize(
    "text",
    ["не является окончательной позицией", "is not the final position of the auditor"],
)
def test_draft_is_never_read_as_final(text: str) -> None:
    """`is not the final position` содержит в себе `the final position`.

    При неверном порядке проверок промежуточная ведомость опозналась бы как
    окончательный отчёт, и её отменённые выводы попали бы в расчёт.
    """
    marked = squeeze(f"ОТЧЁТ О ВЫПОЛНЕНИИ СОГЛАСОВАННЫХ ПРОЦЕДУР {text}")
    assert detect_status(marked, DocumentKind.AUDIT_PROCEDURES) is DocumentStatus.DRAFT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Организация\nEkibastuz Power Services JSC\nСчёт\nACC-7805",
            "Ekibastuz Power Services JSC",
        ),
        ("Entity\nEkibastuz Power Services JSC\nAccount\nACC-7805", "Ekibastuz Power Services JSC"),
    ],
)
def test_organisation_is_language_neutral(text: str, expected: str) -> None:
    assert detect_organisation(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "изложен в отчёте о выполнении согласованных процедур № AR-2025-0634",
        "set out in the agreed-upon procedures report No. AR-2025-0634",
    ],
)
def test_report_reference_is_language_neutral(text: str) -> None:
    assert referenced_report(text) == "AR-2025-0634"


# --- досье KYC ----------------------------------------------------------------

KYC_RU = """
Доля голосующих прав
Aktau Holdings LLP
35.0%
Kaspi Mining LLP
18.7%
Организации, в которых Группа владеет 20.0% и более голосующих прав, признаются
связанными сторонами.
"""

KYC_EN = """
Share of voting rights
Aktau Holdings LLP
35.0%
Kaspi Mining LLP
18.7%
Entities in which the Group holds 20.0% or more of voting rights are treated as
related parties.
"""


def test_kyc_policies_agree_across_languages() -> None:
    russian = parse_related_party_policy(KYC_RU)
    english = parse_related_party_policy(KYC_EN)
    assert russian == english
    assert russian.threshold == Decimal("0.20")
    assert russian.related_parties == ("Aktau Holdings LLP",)


COLLATERAL_RU = """
Доля активов в залоге
Aktau Energy LLP
12.0%
Дочерние организации, у которых доля активов в залоге ниже 25.0%, в периметр
обеспечения не входят.
"""

COLLATERAL_EN = """
Share of pledged assets
Aktau Energy LLP
12.0%
Subsidiaries whose share of pledged assets is below 25.0% are outside the security
perimeter.
"""


def test_collateral_policies_agree_across_languages() -> None:
    assert parse_collateral_policy(COLLATERAL_RU) == parse_collateral_policy(COLLATERAL_EN)


# --- раскрытия ----------------------------------------------------------------


def test_reclassification_by_transaction_agrees() -> None:
    russian = notes.reclassification(
        item(
            "Операция TXN-P1-0007, первоначально учтённая как Прочие расходы "
            "($120,000.00), переклассифицирована для целей соблюдения ковенантов "
            "как Операционные расходы."
        )
    )
    english = notes.reclassification(
        item(
            "Transaction TXN-P1-0007, originally recorded as Other expenses "
            "($120,000.00), has been reclassified for covenant purposes as "
            "Operating costs."
        )
    )
    assert russian is not None and english is not None
    assert russian.txn_id == english.txn_id == "TXN-P1-0007"
    assert russian.amount == english.amount == Money.from_decimal(120000, Currency.USD)
    assert russian.accepted is english.accepted is True


def test_reclassification_by_amount_agrees() -> None:
    russian = notes.reclassification(
        item(
            "Сумма в размере $95,000.00, выплаченная контрагенту Northwind Catering, "
            "первоначально учтённая как Прочее, переклассифицирована для целей "
            "соблюдения ковенантов как Аренда."
        )
    )
    english = notes.reclassification(
        item(
            "An amount of $95,000.00 paid to Northwind Catering, originally recorded "
            "as Other, was reclassified for covenant purposes as Rent."
        )
    )
    assert russian is not None and english is not None
    assert russian.counterparty == english.counterparty == "Northwind Catering"
    assert russian.amount == english.amount


def test_rejected_reclassification_agrees() -> None:
    russian = notes.reclassification(
        item(
            "Операция TXN-P2-0011, первоначально учтённая как Аренда ($50,000.00), "
            "рассматривалась на предмет возможной переклассификации как Прочее; "
            "по итогам рассмотрения первоначальная классификация сохранена."
        )
    )
    english = notes.reclassification(
        item(
            "Transaction TXN-P2-0011, originally recorded as Rent ($50,000.00), was "
            "considered for reclassification as Other; following the review the "
            "original classification is retained."
        )
    )
    assert russian is not None and english is not None
    assert russian.accepted is english.accepted is False
    assert russian.txn_id == english.txn_id


def test_reviewed_no_change_agrees() -> None:
    russian = notes.reclassification(
        item(
            "Операция TXN-P3-0002 ($1,000.00, Northwind Catering) была запрошена "
            "кредитором и проверена; корректировка классификации не требуется."
        )
    )
    english = notes.reclassification(
        item(
            "Transaction TXN-P3-0002 ($1,000.00, Northwind Catering) was requested by "
            "the lender and reviewed; no adjustment to the classification is required."
        )
    )
    assert russian is not None and english is not None
    assert russian.new_value is english.new_value is None
    assert russian.accepted is english.accepted is False


@pytest.mark.parametrize(
    ("russian", "english"),
    [
        (
            "Операция TXN-P7-0033 (Northwind Catering): сумма не отражена в выгрузке "
            "реестра; фактическая сумма операции составляет $250,000.00 (расход)",
            "Transaction TXN-P7-0033 (Northwind Catering): the amount is not reflected "
            "in the ledger extract; the actual amount is $250,000.00 (outflow)",
        ),
    ],
)
def test_missing_amount_keeps_its_sign(russian: str, english: str) -> None:
    assert notes.missing_amount(item(russian)) == notes.missing_amount(item(english))
    found = notes.missing_amount(item(english))
    assert found is not None and found[1].minor < 0


def test_missing_amount_inflow_is_positive() -> None:
    found = notes.missing_amount(
        item(
            "Transaction TXN-P7-0034 (Northwind Catering): the amount is not reflected "
            "in the ledger extract; the actual amount is $250,000.00 (inflow)"
        )
    )
    assert found is not None and found[1].minor > 0


def test_excluded_transaction_agrees() -> None:
    russian = "Операция TXN-P4-0009, датированная 2025-01-05, исключена из ковенантного периода"
    english = "Transaction TXN-P4-0009, dated 2025-01-05, is excluded from the covenant period"
    assert notes.excluded_transaction(item(russian)) == notes.excluded_transaction(item(english))
    assert notes.excluded_transaction(item(english)) == "TXN-P4-0009"


def test_effective_period_agrees() -> None:
    russian = (
        "Операция TXN-P6-0021 (счёт-фактура от 2025-01-10) относится к услугам, "
        "оказанным в период с 2024-11-01"
    )
    english = (
        "Transaction TXN-P6-0021 (invoice dated 2025-01-10) relates to services "
        "rendered in the period from 2024-11-01"
    )
    assert notes.effective_period(item(russian)) == notes.effective_period(item(english))
    assert notes.effective_period(item(english)) == ("TXN-P6-0021", date(2024, 11, 1))


def test_aggregate_obligation_agrees() -> None:
    russian = notes.aggregate_obligation(
        item("совокупное обязательство по программе выходных пособий в размере $400,000.00")
    )
    english = notes.aggregate_obligation(
        item("an aggregate obligation in respect of the severance programme of $400,000.00")
    )
    assert russian is not None and english is not None
    assert russian[1] == english[1] == Money.from_decimal(400000, Currency.USD)


def test_fx_settlement_agrees() -> None:
    russian = notes.fx_settlement(
        item(
            "Расчёты с контрагентом «Ertis Capital»: счёт на сумму 100,000.00 EUR "
            "урегулирован платежом в долларах США в размере $110,000.00"
        )
    )
    english = notes.fx_settlement(
        item(
            'Settlements with counterparty "Ertis Capital": an invoice of 100,000.00 EUR '
            "was settled by a payment in US dollars of $110,000.00"
        )
    )
    assert russian == english
    assert russian is not None and russian[0] == "Ertis Capital"


def test_one_off_policy_agrees() -> None:
    russian = notes.one_off_minimum(
        "Разовыми для целей ковенантов признаются статьи в сумме не менее $300,000.00"
    )
    english = notes.one_off_minimum(
        "Items of not less than $300,000.00 are treated as one-off for covenant purposes"
    )
    assert russian == english == Money.from_decimal(300000, Currency.USD)


def test_reason_label_is_language_neutral() -> None:
    assert item("Что-то произошло. Основание: договор.").reason == "договор."
    assert item("Something happened. Basis: the agreement.").reason == "the agreement."


def test_deferral_to_the_report_is_a_known_shape() -> None:
    """Пункт называет сумму, но вывод оставляет отчёту — это разобрано, а не пропущено."""
    russian = item(
        "Сумма в размере $592,296.10 отобрана для проверки. Вывод изложен в отчёте "
        "о выполнении согласованных процедур № AR-2025-0634 и в настоящих "
        "примечаниях не повторяется."
    )
    english = item(
        "An amount of $592,296.10 was selected for review. The conclusion is set out "
        "in the agreed-upon procedures report No. AR-2025-0634 and is not repeated "
        "in these notes."
    )
    assert notes.is_recognised(russian) and notes.is_recognised(english)


def test_unknown_actionable_disclosure_is_reported() -> None:
    """Незнакомая формулировка с суммой обязана быть видна, а не исчезнуть."""
    unknown = item("Транзакция TXN-P1-0001 на сумму $10,000.00 обработана особым образом.")
    assert notes.is_actionable(unknown)
    assert not notes.is_recognised(unknown)


def test_narrative_paragraph_is_not_reported() -> None:
    """Повествование без суммы и без операции отчёт засорять не должно."""
    narrative = item("Учётная политика применяется последовательно из периода в период.")
    assert not notes.is_actionable(narrative)


# --- движение основных средств ------------------------------------------------


def test_ppe_movement_agrees_across_languages() -> None:
    russian = parse_ppe_movement(
        """
        Основные средства
        Выбытий основных средств не было.
        Балансовая стоимость на начало года
        $100,000.00
        Амортизация за год
        $10,000.00
        Балансовая стоимость на конец года
        $120,000.00
        """
    )
    english = parse_ppe_movement(
        """
        Note 7 — Property, Plant and Equipment
        There were no disposals during the year.
        Net book value at the beginning of the year
        $100,000.00
        Depreciation charge for the year
        $10,000.00
        Net book value at the end of the year
        $120,000.00
        """
    )
    assert russian == english
    assert russian.additions == Decimal("30000.00")


@pytest.mark.parametrize(
    "written",
    ["2024-11-01", "1 November 2024", "November 1, 2024", "November 1 2024"],
)
def test_dates_are_read_in_both_notations(written: str) -> None:
    """ISO и английская пропись дают одну дату: дальше по коду разницы быть не должно."""
    found = notes.effective_period(
        item(
            f"Transaction TXN-P6-0021 (invoice dated 2025-01-10) relates to services "
            f"rendered in the period from {written}"
        )
    )
    assert found == ("TXN-P6-0021", date(2024, 11, 1))


def test_kyc_markdown_table_is_read_in_english() -> None:
    """Из распознавания таблица приходит строками Markdown, а не колонкой."""
    policy = parse_related_party_policy(
        "Share of voting rights\n"
        "| Entity | Share |\n"
        "| Aktau Holdings LLP | 35.0% |\n"
        "| Kaspi Mining LLP | 18.7% |\n"
        "Entities in which the Group holds 20.0% or more of voting rights are treated "
        "as related parties.\n"
    )
    assert policy.threshold == Decimal("0.20")
    assert policy.related_parties == ("Aktau Holdings LLP",)


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        ("Операционные расходы", "opex"),
        ("Operating costs", "opex"),
        ("Interest expense", "interest_expense"),
        ("Capital expenditure", "capex"),
        ("Payroll", "payroll"),
        ("Revenue", "revenue"),
    ],
)
def test_document_stated_category_is_language_neutral(stated: str, expected: str) -> None:
    """Категорию, названную документом, надо понять — иначе операция станет нерешённой."""
    match = next(
        (category for phrase, category in _STATED_CATEGORIES if stated.lower().startswith(phrase)),
        None,
    )
    assert match is not None and match.value == expected
