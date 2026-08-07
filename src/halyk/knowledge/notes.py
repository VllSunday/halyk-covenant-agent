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
from datetime import date

from halyk.money import Currency, Money, parse_money

# Пункт раскрытия: «(9.1) ...» в примечаниях и отчётах, «(1) ...» в записке казначейства.
_ITEM = re.compile(r"\((?P<number>\d+(?:\.\d+)?)\)\s*(?P<body>.*?)(?=\(\d+(?:\.\d+)?\)|$)", re.S)

# Колонтитулы приклеиваются к последнему пункту страницы. Обрезаем по ним, иначе в
# основании корректировки окажется подпись аудитора.
_FOOTER = re.compile(
    r"\s*(?:За аудитора|ПРИМЕЧАНИЯ К ФИНАНСОВОЙ|ОТЧЁТ О ВЫПОЛНЕНИИ|"
    r"ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ|СЛУЖЕБНАЯ ЗАПИСКА|ДОСЬЕ KYC|ДОСЬЕ КУС)"
)

_AMOUNT = r"\$\s?[\d][\d,]*(?:\.\d{2})?"
_TXN = r"TXN-[A-Z0-9]+-\d+"

_BY_AMOUNT = re.compile(
    rf"Сумма в размере (?P<amount>{_AMOUNT}), выплаченная контрагенту (?P<counterparty>.+?), "
    r"первоначально учтённая как (?P<old>.+?), переклассифицирована для целей "
    r"соблюдения ковенантов как (?P<new>.+?)\."
)
_BY_TXN = re.compile(
    rf"Операция (?P<txn_id>{_TXN}), первоначально учтённая как (?P<old>.+?) "
    rf"\((?P<amount>{_AMOUNT})\), переклассифицирована для целей соблюдения "
    r"ковенантов как (?P<new>.+?)\."
)
_REJECTED = re.compile(
    rf"Операция (?P<txn_id>{_TXN}), первоначально учтённая как (?P<old>.+?) "
    rf"\((?P<amount>{_AMOUNT})\), рассматривалась на предмет возможной "
    r"переклассификации как (?P<new>.+?); по итогам .+? первоначальная классификация"
)
_REVIEWED = re.compile(
    rf"Операция (?P<txn_id>{_TXN}) \((?P<amount>{_AMOUNT}), (?P<counterparty>.+?)\) была "
    r"запрошена кредитором и проверена; корректировка .+? не требуется"
)
_MISSING_AMOUNT = re.compile(
    rf"Операция (?P<txn_id>{_TXN}) \((?P<counterparty>.+?)\): сумма не отражена в выгрузке "
    rf"реестра; фактическая сумма операции составляет (?P<amount>{_AMOUNT})\s*"
    r"\((?P<direction>расход|поступление)\)"
)
_EXCLUDED = re.compile(
    rf"Операция (?P<txn_id>{_TXN}), датированная (?P<date>\d{{4}}-\d{{2}}-\d{{2}}), "
    r"исключена из ковенантного периода"
)
_RENDERED_IN = re.compile(
    rf"Операция (?P<txn_id>{_TXN}) \(счёт-фактура от (?P<invoiced>\d{{4}}-\d{{2}}-\d{{2}})\) "
    r"относится к услугам, оказанным в период с (?P<start>\d{4}-\d{2}-\d{2})"
)
_OBLIGATION = re.compile(
    rf"совокупное обязательство по (?P<description>.+?) в размере (?P<amount>{_AMOUNT})"
)
_FX_SETTLEMENT = re.compile(
    r"Расчёты с контрагентом «(?P<counterparty>.+?)»: счёт на сумму "
    r"(?P<invoiced>[\d,]+\.\d{2})\s*(?P<currency>EUR|KZT|USD) урегулирован платежом "
    rf"в долларах США в размере (?P<settled>{_AMOUNT})"
)
_ONE_OFF_POLICY = re.compile(
    rf"Разовыми для целей ковенантов признаются статьи в сумме не менее (?P<amount>{_AMOUNT})"
)
_ONE_OFF_ROW = re.compile(
    rf"^\|\s*(?P<description>[^|\n]+?)\s*\|\s*(?P<counterparty>[^|\n]+?)\s*\|\s*"
    rf"(?P<amount>{_AMOUNT})\s*\|",
    re.M,
)
_REASON = re.compile(r"Основание:\s*(?P<reason>.+)", re.S)


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
    match = _ONE_OFF_POLICY.search(flatten(page_text))
    return _money(match.group("amount")) if match else None


def aggregate_obligation(item: Disclosure) -> tuple[str, Money] | None:
    match = _OBLIGATION.search(item.body)
    if match is None:
        return None
    return match.group("description").strip(), _money(match.group("amount"))


def fx_settlement(item: Disclosure) -> tuple[str, Money, Money] | None:
    match = _FX_SETTLEMENT.search(item.body)
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
    if (match := _REJECTED.search(item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            txn_id=match.group("txn_id"),
            accepted=False,
        )
    if (match := _BY_TXN.search(item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            txn_id=match.group("txn_id"),
        )
    if (match := _BY_AMOUNT.search(item.body)) is not None:
        return Reclassification(
            old_value=match.group("old").strip(),
            new_value=match.group("new").strip(),
            amount=_money(match.group("amount")),
            counterparty=match.group("counterparty").strip(),
        )
    if (match := _REVIEWED.search(item.body)) is not None:
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
    match = _MISSING_AMOUNT.search(item.body)
    if match is None:
        return None
    amount = _money(match.group("amount"))
    return match.group("txn_id"), -amount if match.group("direction") == "расход" else amount


def excluded_transaction(item: Disclosure) -> str | None:
    match = _EXCLUDED.search(item.body)
    return match.group("txn_id") if match else None


def effective_period(item: Disclosure) -> tuple[str, date] | None:
    """Операция, услуги по которой оказаны в другом периоде."""
    match = _RENDERED_IN.search(item.body)
    if match is None:
        return None
    return match.group("txn_id"), date.fromisoformat(match.group("start"))
