"""Разбор ковенантных раскрытий. Формулировки взяты из документов открытого набора."""

from datetime import date
from decimal import Decimal

from halyk.knowledge import notes
from halyk.money import Currency, Money

RECLASS_BY_AMOUNT = """Примечание 9 — Переклассификации для целей соблюдения ковенантов
(9.1) Сумма в размере $592,296.10, выплаченная контрагенту Irtysh Advisory Bureau,
первоначально учтённая как Консультационные услуги, переклассифицирована для целей
соблюдения ковенантов как Процентные расходы.
Основание: Вознаграждение по договору по существу является платой за финансирование по
бридж-кредиту и для целей ковенантов учитывается как процентные расходы.
За аудитора и от его имени
Altyn-Tau Audit LLP
"""

DRAFT_BY_TXN = """4. Предварительные вопросы по классификации операций
(4.1) Операция TXN-P6-0044, первоначально учтённая как Коммунальные услуги ($418,204.37),
переклассифицирована для целей соблюдения ковенантов как Налоги.
Основание: вопрос поставлен на промежуточном этапе по подтверждающим документам к проводке
ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ ВОПРОСОВ ПО КЛАССИФИКАЦИИ · TARAZ CEMENT WORKS JSC
"""

REJECTED = """(7.2) Операция TXN-P10-0012, первоначально учтённая как Операционные расходы
($118,447.52), рассматривалась на предмет возможной переклассификации как Страховые
премии; по итогам разъяснений руководства первоначальная классификация (Операционные
расходы) сохраняется, и корректировка для целей ковенантов не производилась.
Основание: Работы выполняются подрядчиком Заёмщика вне условий страхового полиса.
"""

REVIEWED = """(8.1) Операция TXN-P10-0021 ($1,204,663.28, Saryarka Terminal Properties LLP)
была запрошена кредитором и проверена; корректировка для целей ковенантов не требуется, и
её первоначальная классификация сохраняется.
"""

MISSING = """Примечание 8 — Суммы, не отражённые в выгрузке реестра
(8.1) Операция TXN-P8-0031 (Kyzylorda Drilling Personnel LLP): сумма не отражена в выгрузке
реестра; фактическая сумма операции составляет $884,204.16 (расход).
"""

EXCLUDED = """Примечание 9 — Отсечение и начисления
(9.1) Операция TXN-B4-0026, датированная 2025-11-20, исключена из ковенантного периода
2025 года.
Основание: Право собственности и риски по грузу переходят только в январе 2026 года.
"""

RENDERED_LATER = """Примечание 7 — Отсечение и начисления
(7.1) Операция TXN-P1-0045 (счёт-фактура от 2025-08-12) относится к услугам, оказанным в
период с 2026-01-15 по 2026-03-20.
Основание: Обследование причальной стенки проводится в первом квартале 2026 года.
"""

OBLIGATION = """Примечание 7 — Раскрытия для агрегирования ковенантов
(7.1) Для целей агрегирования по ковенантам совокупное обязательство по программе выходных
пособий в размере $918,447.52 раскрывается и не отражается отдельной операцией в
бухгалтерской книге.
"""

FX = """Примечание 9 — Валютные курсы
(9.1) Расчёты с контрагентом «Rheinland Katalyse Service GmbH»: счёт на сумму 72,146.75 EUR
урегулирован платежом в долларах США в размере $83,690.23.
"""

ONE_OFF = """Примечание 8 — Корректировки EBITDA

| Характер статьи | Контрагент | Сумма |
|---|---|---|
| Очистка причального дна от наносов | «Zhaiyk Dredging LLP» | $251,338.94 |
| Урегулирование спора по демереджу | «Aral Freight Arbitration Bureau» | $342,905.28 |

Разовыми для целей ковенантов признаются статьи в сумме не менее $300,000.00; статьи
меньшей суммы к EBITDA не прибавляются.
"""


def only(text: str) -> notes.Disclosure:
    items = list(notes.disclosures(text))
    assert len(items) == 1, [item.number for item in items]
    return items[0]


def test_reclassification_by_amount_and_counterparty() -> None:
    """Окончательный отчёт называет сумму и контрагента, но не номер проводки."""
    change = notes.reclassification(only(RECLASS_BY_AMOUNT))
    assert change is not None
    assert change.txn_id is None
    assert change.counterparty == "Irtysh Advisory Bureau"
    assert change.amount == Money.from_decimal(Decimal("592296.10"), Currency.USD)
    assert (change.old_value, change.new_value) == ("Консультационные услуги", "Процентные расходы")
    assert change.accepted


def test_reason_stops_before_the_footer() -> None:
    # Иначе подпись аудитора попадёт в объяснение корректировки.
    reason = only(RECLASS_BY_AMOUNT).reason
    assert reason.endswith("учитывается как процентные расходы.")
    assert "Altyn-Tau" not in reason


def test_reclassification_by_transaction_id() -> None:
    change = notes.reclassification(only(DRAFT_BY_TXN))
    assert change is not None
    assert change.txn_id == "TXN-P6-0044"
    assert change.new_value == "Налоги"


def test_considered_and_rejected_is_not_an_accepted_change() -> None:
    """Рассмотренная и отклонённая переклассификация — ловушка датасета.

    Её нужно видеть в отчёте, но применять нельзя: документ прямо сохраняет
    первоначальную классификацию.
    """
    change = notes.reclassification(only(REJECTED))
    assert change is not None
    assert change.txn_id == "TXN-P10-0012"
    assert not change.accepted


def test_reviewed_without_change_is_recorded() -> None:
    change = notes.reclassification(only(REVIEWED))
    assert change is not None
    assert change.new_value is None
    assert not change.accepted


def test_missing_amount_takes_its_sign_from_the_direction() -> None:
    found = notes.missing_amount(only(MISSING))
    assert found == ("TXN-P8-0031", Money.from_decimal(Decimal("-884204.16"), Currency.USD))


def test_excluded_transaction() -> None:
    assert notes.excluded_transaction(only(EXCLUDED)) == "TXN-B4-0026"


def test_effective_period_uses_the_start_of_service() -> None:
    assert notes.effective_period(only(RENDERED_LATER)) == ("TXN-P1-0045", date(2026, 1, 15))


def test_aggregate_obligation() -> None:
    found = notes.aggregate_obligation(only(OBLIGATION))
    assert found is not None
    description, amount = found
    assert description == "программе выходных пособий"
    assert amount == Money.from_decimal(Decimal("918447.52"), Currency.USD)


def test_fx_settlement_keeps_both_amounts() -> None:
    found = notes.fx_settlement(only(FX))
    assert found is not None
    counterparty, invoiced, settled = found
    assert counterparty == "Rheinland Katalyse Service GmbH"
    assert invoiced == Money.from_decimal(Decimal("72146.75"), Currency.EUR)
    assert settled == Money.from_decimal(Decimal("83690.23"), Currency.USD)


def test_one_off_items_are_read_from_the_table() -> None:
    items = notes.one_off_items(ONE_OFF)
    assert [description for description, _, _ in items] == [
        "Очистка причального дна от наносов",
        "Урегулирование спора по демереджу",
    ]
    assert items[0][1] == "Zhaiyk Dredging LLP"
    assert items[0][2] == Money.from_decimal(Decimal("251338.94"), Currency.USD)


def test_one_off_minimum() -> None:
    assert notes.one_off_minimum(ONE_OFF) == Money.from_decimal(Decimal("300000"), Currency.USD)


def test_disclosures_keep_their_own_reasons() -> None:
    """У каждого пункта своё основание, и склеивать их нельзя."""
    items = list(notes.disclosures(REJECTED + REVIEWED))
    assert [item.number for item in items] == ["7.2", "8.1"]
    assert "страхового полиса" in items[0].reason
    assert items[1].reason == ""


def test_transaction_id_survives_extra_segments_and_cyrillic_homoglyphs() -> None:
    """Распознавание русской страницы возвращает идентификатор кириллицей.

    На вид он тот же, но в реестре по нему ничего не находится, и раскрытие про
    эту операцию молча не доходит до расчёта.
    """
    item = notes.Disclosure(
        number="9.1",
        body=(
            "Операция TXN-КС-МКТ-05, первоначально учтённая как Маркетинговые расходы "
            "($20,284,662.18), переклассифицирована для целей соблюдения ковенантов "
            "как Капитальные затраты."
        ),
    )
    change = notes.reclassification(item)
    assert change is not None
    assert change.txn_id == "TXN-KC-MKT-05"


def test_transaction_included_into_period() -> None:
    item = notes.Disclosure(
        number="4.1",
        body=(
            "Операция TXN-S2-0010, датированная 2026-01-05, включена в ковенантный "
            "период 2025 года."
        ),
    )
    assert notes.included_transaction(item) == "TXN-S2-0010"
    assert notes.excluded_transaction(item) is None


def test_bank_charge_is_added_back_to_the_converted_amount() -> None:
    """Платёж назван за вычетом комиссии, а комиссия в пересчёт не входит.

    Оставив её вычтенной, курс вберёт в себя стоимость перевода и исказит
    пересчёт всех операций контрагента.
    """
    item = notes.Disclosure(
        number="4.1",
        body=(
            "Settlement with Donau Metallhandel GmbH: an invoice of 57,338.50 EUR was "
            "settled by a payment of $64,322.63, stated net of a correspondent bank "
            "charge of $1,043.26, which does not form part of the converted amount."
        ),
    )
    settlement = notes.fx_settlement(item)
    assert settlement is not None
    counterparty, invoiced, settled = settlement
    assert counterparty == "Donau Metallhandel GmbH"
    assert invoiced.to_decimal() == Decimal("57338.50")
    assert settled.to_decimal() == Decimal("65365.89")
