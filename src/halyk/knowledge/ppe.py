"""Движение основных средств: восстановление капитальных затрат из отчётности.

Отдельной строки «capital expenditure» в консолидированном отчёте нет. Величина
выводится из тождества движения основных средств, и это единственный способ её
получить.

Тождество замыкается только тогда, когда раскрыты все движения. Поэтому парсер
собирает их поимённо и отказывается считать, если в разделе встретилась строка,
которой он не знает: приобретение дочерних компаний, обесценение, переоценка или
курсовая разница меняют равенство, и молча пропущенная строка даст правдоподобное,
но неверное число.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

_AMOUNT = r"\$?\s*(?P<amount>-?[\d  , ]+\.\d{2})"

# Названия строк раскрытия. Каждое ведёт себя в тождестве по-своему, поэтому список
# закрыт: незнакомая строка — повод отказаться, а не проигнорировать её.
_MOVEMENTS: tuple[tuple[str, str], ...] = (
    ("opening", r"net book value at the beginning of the year|на начало года"),
    ("closing", r"net book value at the end of the year|на конец года"),
    ("depreciation", r"depreciation charge for the year|амортизац\w*"),
    ("disposals", r"disposals?|выбыти\w*"),
    ("additions", r"additions?|поступлени\w*|приобретени\w*"),
    ("impairment", r"impairment\w*|обесценени\w*"),
    ("revaluation", r"revaluation\w*|переоценк\w*"),
    ("acquisitions", r"acquisition of subsidiar\w*|приобретение дочерн\w*"),
    ("exchange", r"exchange differences?|курсов\w* разниц\w*"),
    ("transfers", r"transfers?|перевод\w* между"),
)

_SECTION = re.compile(
    r"(?:Note\s*\d+\s*[—-]\s*)?Property,?\s+Plant\s+and\s+Equipment|"
    r"Основные\s+средства",
    re.IGNORECASE,
)
_NO_DISPOSALS = re.compile(
    r"(?:there\s+were\s+)?no\s+disposals|выбыти\w*[^.\n]*?(?:не\s+было|отсутств\w*)",
    re.IGNORECASE,
)


class PpeError(ValueError):
    """Движение основных средств прочитать нельзя. Догадка здесь дороже отказа."""


@dataclass(frozen=True, slots=True)
class PpeMovement:
    """Раскрытые движения основных средств за период."""

    opening: Decimal
    closing: Decimal
    depreciation: Decimal
    disposals: Decimal = Decimal(0)
    other: Decimal = Decimal(0)

    @property
    def additions(self) -> Decimal:
        """Капитальные затраты из тождества движения.

        `closing = opening + additions − depreciation − disposals + other`, отсюда
        `additions = closing − opening + depreciation + disposals − other`.
        """
        return self.closing - self.opening + self.depreciation + self.disposals - self.other


def _amount(raw: str) -> Decimal:
    return Decimal(re.sub(r"[  , ]", "", raw))


def _find(text: str, pattern: str) -> Decimal | None:
    """Число, стоящее за названием строки.

    Значение может идти и на той же строке, и на следующей: в текстовом слое PDF
    таблица приходит колонкой, а из OCR — строкой Markdown.

    Между названием и числом допускается хвост той же строки («Impairment loss»,
    «Revaluation surplus»), но не перевод строки сверх одного: иначе название без
    своего числа притянуло бы значение соседней строки таблицы.
    """
    match = re.search(rf"(?:{pattern})[^\n]*?\s*[:|]?\s*\n?\s*{_AMOUNT}", text, re.IGNORECASE)
    return _amount(match.group("amount")) if match else None


def parse_ppe_movement(text: str) -> PpeMovement:
    """Разбор раздела об основных средствах.

    Отказ вместо догадки везде, где тождество может не замкнуться: нет раздела, нет
    одной из трёх обязательных величин, встретилось движение, которого мы не умеем
    учитывать.
    """
    if _SECTION.search(text) is None:
        raise PpeError("В документе нет раздела о движении основных средств")

    found = {name: _find(text, pattern) for name, pattern in _MOVEMENTS}
    required = {}
    for name in ("opening", "closing", "depreciation"):
        if (value := found[name]) is None:
            raise PpeError(f"В разделе не раскрыта обязательная величина: {name}")
        required[name] = value

    # Выбытий может не быть вовсе, и об этом сказано текстом. Отличать «раскрыто как
    # ноль» от «не раскрыто» обязательно: во втором случае тождество не замыкается.
    disposals = found["disposals"]
    if disposals is None:
        if _NO_DISPOSALS.search(text) is None:
            raise PpeError("Выбытия не раскрыты и не объявлены отсутствующими")
        disposals = Decimal(0)

    unexpected = [
        name
        for name in (
            "additions",
            "impairment",
            "revaluation",
            "acquisitions",
            "exchange",
            "transfers",
        )
        if found[name] is not None
    ]
    if unexpected:
        # Прямо раскрытые поступления — не проблема, а подарок: считать их из
        # тождества незачем. Остальные движения меняют равенство, и пока они не
        # разобраны поимённо, выводить капзатраты нельзя.
        beyond = [name for name in unexpected if name != "additions"]
        if beyond:
            raise PpeError(
                f"В разделе раскрыты движения, которые мы не учитываем: {', '.join(beyond)}"
            )

    movement = PpeMovement(
        opening=required["opening"],
        closing=required["closing"],
        depreciation=required["depreciation"],
        disposals=disposals,
    )

    if (declared := found["additions"]) is not None and declared != movement.additions:
        raise PpeError(
            f"Раскрытые поступления {declared} расходятся с выведенными "
            f"{movement.additions}: тождество движения не замыкается"
        )
    return movement
