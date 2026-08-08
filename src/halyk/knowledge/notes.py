"""Разбор ковенантных раскрытий: примечаний, отчётов аудитора и записки казначейства.

Раскрытия пронумерованы пунктами, и разбор идёт по пунктам, а не по всей странице:
у каждого пункта своё основание, и склеивать их нельзя — иначе причина одной
корректировки попадёт в объяснение другой.

Текст читается детерминированно, регулярными выражениями. Формулировки в датасете
шаблонные, а цена ошибки здесь выше, чем выигрыш от гибкости: подставленная моделью
сумма выглядит так же убедительно, как настоящая.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime

from halyk.money import Currency, Money, parse_money

# Пункт раскрытия: «(9.1) ...» в примечаниях и отчётах, «(1) ...» в записке казначейства.
_ITEM = re.compile(r"\((?P<number>\d+(?:\.\d+)?)\)\s*(?P<body>.*?)(?=\(\d+(?:\.\d+)?\)|$)", re.S)

# Колонтитулы приклеиваются к последнему пункту страницы. Обрезаем по ним, иначе в
# основании корректировки окажется подпись аудитора.
_FOOTER = re.compile(
    r"\s*(?:За аудитора|ПРИМЕЧАНИЯ К ФИНАНСОВОЙ|ОТЧЁТ О ВЫПОЛНЕНИИ|"
    r"ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ|СЛУЖЕБНАЯ ЗАПИСКА|ДОСЬЕ KYC|ДОСЬЕ КУС|"
    r"For the auditor|NOTES TO THE FINANCIAL|AGREED[- ]UPON PROCEDURES|"
    r"INTERIM SCHEDULE|TREASURY MEMORANDUM|KYC FILE)"
)

_AMOUNT = r"\$\s?[\d][\d,]*(?:\.\d{2})?"
_TXN = r"TXN-[A-Z0-9]+-\d+"
# Дата в раскрытиях приходит в ISO, но англоязычный документ может написать её
# прописью. Оба начертания разбираются одной функцией, чтобы дальше по коду разницы
# не было.
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_DATE = (
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+\d{4}"
    r"|(?:" + "|".join(_MONTHS) + r")\s+\d{1,2},?\s+\d{4}"
)

# Каждое раскрытие описано парой шаблонов: русским и английским. Именованные группы у
# них одинаковые, поэтому дальше по коду язык уже неразличим — обе ветви дают один и
# тот же Fact или Adjustment. Отдельной английской модели нет намеренно: две модели
# разошлись бы на первой же правке.
_BY_AMOUNT = (
    re.compile(
        rf"Сумма в размере (?P<amount>{_AMOUNT}), выплаченная контрагенту (?P<counterparty>.+?), "
        r"первоначально учтённая как (?P<old>.+?), переклассифицирована для целей "
        r"соблюдения ковенантов как (?P<new>.+?)\."
    ),
    re.compile(
        rf"(?:An )?amount of (?P<amount>{_AMOUNT}) paid to (?P<counterparty>.+?), "
        r"originally recorded as (?P<old>.+?), (?:has been |was )?reclassified for "
        r"covenant purposes as (?P<new>.+?)\."
    ),
)
_BY_TXN = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}), первоначально учтённая как (?P<old>.+?) "
        rf"\((?P<amount>{_AMOUNT})\), переклассифицирована для целей соблюдения "
        r"ковенантов как (?P<new>.+?)\."
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}), originally recorded as (?P<old>.+?) "
        rf"\((?P<amount>{_AMOUNT})\), (?:has been |was )?reclassified for covenant "
        r"purposes as (?P<new>.+?)\."
    ),
)
_REJECTED = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}), первоначально учтённая как (?P<old>.+?) "
        rf"\((?P<amount>{_AMOUNT})\), рассматривалась на предмет возможной "
        r"переклассификации как (?P<new>.+?); по итогам .+? первоначальная классификация"
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}), originally recorded as (?P<old>.+?) "
        rf"\((?P<amount>{_AMOUNT})\), was considered for reclassification as "
        r"(?P<new>.+?); (?:following|after) .+? the original classification"
    ),
)
_REVIEWED = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}) \((?P<amount>{_AMOUNT}), (?P<counterparty>.+?)\) была "
        r"запрошена кредитором и проверена; корректировка .+? не требуется"
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}) \((?P<amount>{_AMOUNT}), (?P<counterparty>.+?)\) was "
        r"requested by the lender and reviewed; no adjustment .+? (?:is )?required"
    ),
)
_MISSING_AMOUNT = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}) \((?P<counterparty>.+?)\): сумма не отражена в выгрузке "
        rf"реестра; фактическая сумма операции составляет (?P<amount>{_AMOUNT})\s*"
        r"\((?P<direction>расход|поступление)\)"
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}) \((?P<counterparty>.+?)\): the amount is not "
        rf"reflected in the ledger extract; the actual amount is (?P<amount>{_AMOUNT})\s*"
        r"\((?P<direction>outflow|inflow|expense|receipt)\)"
    ),
)
_EXCLUDED = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}), датированная (?P<date>{_DATE}), "
        r"исключена из ковенантного периода"
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}), dated (?P<date>{_DATE}), "
        r"is excluded from the covenant period"
    ),
)
_RENDERED_IN = (
    re.compile(
        rf"Операция (?P<txn_id>{_TXN}) \(счёт-фактура от (?P<invoiced>{_DATE})\) "
        rf"относится к услугам, оказанным в период с (?P<start>{_DATE})"
    ),
    re.compile(
        rf"Transaction (?P<txn_id>{_TXN}) \(invoice dated (?P<invoiced>{_DATE})\) "
        rf"relates to services rendered in the period from (?P<start>{_DATE})"
    ),
)
_OBLIGATION = (
    re.compile(
        rf"совокупное обязательство по (?P<description>.+?) в размере (?P<amount>{_AMOUNT})"
    ),
    re.compile(
        rf"(?:an )?aggregate obligation (?:in respect of |for |under )(?P<description>.+?) "
        rf"of (?P<amount>{_AMOUNT})"
    ),
)
_FX_SETTLEMENT = (
    re.compile(
        r"Расчёты с контрагентом «(?P<counterparty>.+?)»: счёт на сумму "
        r"(?P<invoiced>[\d,]+\.\d{2})\s*(?P<currency>EUR|KZT|USD) урегулирован платежом "
        rf"в долларах США в размере (?P<settled>{_AMOUNT})"
    ),
    re.compile(
        r"Settlements with (?:counterparty )?[«\"](?P<counterparty>.+?)[»\"]: an invoice of "
        r"(?P<invoiced>[\d,]+\.\d{2})\s*(?P<currency>EUR|KZT|USD) was settled by a payment "
        rf"in US dollars of (?P<settled>{_AMOUNT})"
    ),
)
_ONE_OFF_POLICY = (
    re.compile(
        rf"Разовыми для целей ковенантов признаются статьи в сумме не менее (?P<amount>{_AMOUNT})"
    ),
    re.compile(
        r"[Ii]tems of (?:not less than|at least) (?P<amount>"
        rf"{_AMOUNT}) are treated as one-off for covenant purposes"
    ),
)
# Направление в русском и английском написано по-разному, а знак суммы от него зависит.
_OUTFLOW_WORDS = frozenset({"расход", "outflow", "expense"})
_ONE_OFF_ROW = re.compile(
    rf"^\|\s*(?P<description>[^|\n]+?)\s*\|\s*(?P<counterparty>[^|\n]+?)\s*\|\s*"
    rf"(?P<amount>{_AMOUNT})\s*\|",
    re.M,
)
_REASON = re.compile(r"(?:Основание|Basis|Reason|Grounds):\s*(?P<reason>.+)", re.S)

# Признак того, что пункт что-то предписывает: идентификатор операции или сумма.
# Нужен, чтобы отличить непрочитанное раскрытие от повествовательного абзаца —
# первое обязано попасть в отчёт, второе не должно его засорять.
_ACTIONABLE = re.compile(rf"{_TXN}|{_AMOUNT}")

# Отсылка к другому документу: примечания называют сумму, но вывод по ней делает
# отчёт аудитора. Такой пункт корректировки не порождает — и это разобранный случай,
# а не пропущенный, поэтому он объявлен явно.
_DEFERRED = (
    re.compile(r"изложен в отчёте .+? и в настоящих примечаниях не повторяется"),
    re.compile(r"(?:is )?set out in the .+? and is not repeated in these notes", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class Disclosure:
    """Один пункт раскрытия: номер, текст и основание, если оно приведено."""

    number: str
    body: str

    @property
    def reason(self) -> str:
        match = _REASON.search(self.body)
        return _cut_footer(match.group("reason")) if match else ""


def flatten(text: str) -> str:
    """Схлопывает переносы: раскрытие разорвано ими посреди фразы."""
    return re.sub(r"[ \t]*\n[ \t]*", " ", text).strip()


def _cut_footer(text: str) -> str:
    return _FOOTER.split(text, maxsplit=1)[0].strip()


def disclosures(page_text: str) -> Iterator[Disclosure]:
    for match in _ITEM.finditer(flatten(page_text)):
        yield Disclosure(number=match.group("number"), body=_cut_footer(match.group("body")))


def _money(raw: str) -> Money:
    return parse_money(raw)


def parse_date(raw: str) -> date:
    """Дата раскрытия: ISO или английская пропись.

    Приводится к одному типу здесь, чтобы ни корректировки, ни исполнитель не знали,
    в каком начертании она была напечатана.
    """
    text = raw.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%d %B %Y", "%B %d %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007
        except ValueError:
            continue
    raise ValueError(f"Не разобрал дату {raw!r}")


def _first(patterns: tuple[re.Pattern[str], ...], text: str) -> re.Match[str] | None:
    """Первое совпадение из языковых вариантов шаблона.

    Документ написан на одном языке целиком, поэтому вариантам не за что
    конкурировать: сработает ровно тот, чей язык совпал.
    """
    for pattern in patterns:
        if (match := pattern.search(text)) is not None:
            return match
    return None


def one_off_items(page_text: str) -> tuple[tuple[str, str, Money], ...]:
    """Разовые статьи из таблицы примечаний: назначение, контрагент, сумма."""
    return tuple(
        (
            row.group("description").strip(),
            row.group("counterparty").strip("«» "),
            _money(row.group("amount")),
        )
        for row in _ONE_OFF_ROW.finditer(page_text)
    )


def one_off_minimum(page_text: str) -> Money | None:
    match = _first(_ONE_OFF_POLICY, flatten(page_text))
    return _money(match.group("amount")) if match else None


def aggregate_obligation(item: Disclosure) -> tuple[str, Money] | None:
    match = _first(_OBLIGATION, item.body)
    if match is None:
        return None
    return match.group("description").strip(), _money(match.group("amount"))


def fx_settlement(item: Disclosure) -> tuple[str, Money, Money] | None:
    match = _first(_FX_SETTLEMENT, item.body)
    if match is None:
        return None
    invoiced = parse_money(match.group("invoiced"), Currency(match.group("currency")))
    return match.group("counterparty").strip(), invoiced, _money(match.group("settled"))


@dataclass(frozen=True, slots=True)
class Reclassification:
    """Вывод о классификации операции, как он изложен в документе."""

    old_value: str
    new_value: str | None
    amount: Money | None
    txn_id: str | None = None
    counterparty: str | None = None
    accepted: bool = True


def reclassification(item: Disclosure) -> Reclassification | None:
    """Переклассификация из пункта раскрытия, включая рассмотренную и отклонённую."""
    if (match := _first(_REJECTED, item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            txn_id=match.group("txn_id"),
            accepted=False,
        )
    if (match := _first(_BY_TXN, item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            txn_id=match.group("txn_id"),
        )
    if (match := _first(_BY_AMOUNT, item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            counterparty=match.group("counterparty").strip(),
        )
    if (match := _first(_REVIEWED, item.body)) is not None:
        return Reclassification(
            old_value="",
            new_value=None,
            amount=_money(match.group("amount")),
            txn_id=match.group("txn_id"),
            counterparty=match.group("counterparty").strip(),
            accepted=False,
        )
    return None


def missing_amount(item: Disclosure) -> tuple[str, Money] | None:
    """Сумма, не попавшая в выгрузку. Направление операции задаёт знак."""
    match = _first(_MISSING_AMOUNT, item.body)
    if match is None:
        return None
    amount = _money(match.group("amount"))
    outflow = match.group("direction").lower() in _OUTFLOW_WORDS
    return match.group("txn_id"), -amount if outflow else amount


def excluded_transaction(item: Disclosure) -> str | None:
    match = _first(_EXCLUDED, item.body)
    return match.group("txn_id") if match else None


def effective_period(item: Disclosure) -> tuple[str, date] | None:
    """Операция, услуги по которой оказаны в другом периоде."""
    match = _first(_RENDERED_IN, item.body)
    if match is None:
        return None
    return match.group("txn_id"), parse_date(match.group("start"))


def is_actionable(item: Disclosure) -> bool:
    """Пункт что-то предписывает, а не просто повествует."""
    return _ACTIONABLE.search(item.body) is not None


def is_recognised(item: Disclosure) -> bool:
    """Хотя бы одна семантика разобрала пункт.

    Отрицательный ответ на предписывающем пункте означает непрочитанное раскрытие:
    формулировку изменили или она пришла на языке, которого мы не знаем. Молчаливое
    «фактов нет» здесь неотличимо от «в документе ничего не сказано», а разница
    стоит ячейки.
    """
    return any(
        (
            reclassification(item) is not None,
            missing_amount(item) is not None,
            excluded_transaction(item) is not None,
            effective_period(item) is not None,
            aggregate_obligation(item) is not None,
            fx_settlement(item) is not None,
            one_off_minimum(item.body) is not None,
            defers_to_report(item),
        )
    )


def defers_to_report(item: Disclosure) -> bool:
    """Пункт называет сумму, но вывод по ней оставляет отчёту аудитора."""
    return _first(_DEFERRED, item.body) is not None
