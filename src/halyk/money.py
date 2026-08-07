"""Деньги внутри пайплайна — целое число минорных единиц. Обоснование в docs/adr/0002."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Self


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    KZT = "KZT"


# Множитель хранится явно, а не берётся как 100 по умолчанию: валюта без минорной
# единицы сломала бы арифметику молча, а здесь она упадёт на KeyError.
MINOR_UNITS: dict[Currency, int] = {
    Currency.USD: 100,
    Currency.EUR: 100,
    Currency.KZT: 100,
}

# Порядок важен: «USD» встречается внутри строк вроде «USD 1,000», а «$» — перед
# суммой, поэтому сначала ищем однозначные символы.
_CURRENCY_MARKS: tuple[tuple[str, Currency], ...] = (
    ("$", Currency.USD),
    ("€", Currency.EUR),
    ("₸", Currency.KZT),
    ("USD", Currency.USD),
    ("EUR", Currency.EUR),
    ("KZT", Currency.KZT),
    ("тенге", Currency.KZT),
)

# Множители из текста договоров: «требование на сумму менее 3.75 млн». Английские
# формы нужны потому, что документы датасета двуязычные: названия статей и сумм
# встречаются на обоих языках в одном абзаце.
_MULTIPLIERS: dict[str, int] = {
    "тыс": 1_000,
    "тысяч": 1_000,
    "млн": 1_000_000,
    "миллион": 1_000_000,
    "млрд": 1_000_000_000,
    "миллиард": 1_000_000_000,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_AMOUNT = re.compile(r"(?P<sign>[-−–])?\s*(?P<digits>\d[\d\s  ',.]*\d|\d)")
_MULTIPLIER = re.compile("|".join(sorted(_MULTIPLIERS, key=len, reverse=True)), re.IGNORECASE)
_GROUPING = re.compile(r"[\s  ']")
_DECIMAL_TAIL = re.compile(r"[.,](\d{1,2})$")


class Rounding(StrEnum):
    HALF_UP = "half_up"
    HALF_EVEN = "half_even"

    @property
    def decimal_mode(self) -> str:
        return ROUND_HALF_UP if self is Rounding.HALF_UP else ROUND_HALF_EVEN


class MoneyParseError(ValueError):
    pass


class CurrencyMismatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Money:
    """Сумма в минорных единицах вместе со своей валютой.

    Валюта носится с суммой, а не подразумевается снаружи: в реестре соседствуют USD
    и EUR, и сложение их без пересчёта — ошибка, которую нужно ловить типом, а не
    ревью.
    """

    currency: Currency
    minor: int

    @classmethod
    def from_decimal(cls, value: Decimal | int | str, currency: Currency) -> Self:
        try:
            exact = Decimal(value) * MINOR_UNITS[currency]
        except InvalidOperation as exc:
            raise MoneyParseError(f"Не привёл {value!r} к сумме") from exc
        if exact != exact.to_integral_value():
            raise MoneyParseError(f"{value!r} содержит доли минорной единицы {currency}")
        return cls(currency=currency, minor=int(exact))

    @classmethod
    def zero(cls, currency: Currency) -> Self:
        return cls(currency=currency, minor=0)

    def to_decimal(self) -> Decimal:
        return Decimal(self.minor) / MINOR_UNITS[self.currency]

    def _require_same_currency(self, other: Money) -> None:
        if self.currency is not other.currency:
            raise CurrencyMismatchError(
                f"{self.currency} и {other.currency} нельзя складывать без пересчёта по курсу"
            )

    def __add__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(currency=self.currency, minor=self.minor + other.minor)

    def __sub__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(currency=self.currency, minor=self.minor - other.minor)

    def __neg__(self) -> Money:
        return Money(currency=self.currency, minor=-self.minor)

    def __abs__(self) -> Money:
        return Money(currency=self.currency, minor=abs(self.minor))


def detect_currency(text: str) -> Currency | None:
    positions = [(text.find(mark), currency) for mark, currency in _CURRENCY_MARKS]
    found = [(index, currency) for index, currency in positions if index >= 0]
    return min(found)[1] if found else None


def _split_decimal(digits: str) -> tuple[str, str]:
    """Делит число на целую и дробную часть.

    Десятичным считается последний разделитель, за которым стоят одна-две цифры;
    остальные точки и запятые разрядные. Иначе `$1,500,000.00` из договора и
    `50 000 000,00` из русского текста нельзя разобрать одним правилом.
    """
    cleaned = _GROUPING.sub("", digits)
    if (tail := _DECIMAL_TAIL.search(cleaned)) is not None:
        whole, frac = cleaned[: tail.start()], tail.group(1)
    else:
        whole, frac = cleaned, ""

    whole = whole.replace(".", "").replace(",", "")
    if not whole.isdigit():
        raise MoneyParseError(f"Не разобрал разряды в {digits!r}")
    return whole, frac.ljust(2, "0")


def parse_money(text: str, currency: Currency | None = None) -> Money:
    """Разбирает денежную величину из текста документа.

    Валюта берётся из самого текста, если он её называет, иначе из аргумента.
    Скобочная запись прописью — «50 000 000 (пятьдесят миллионов)» — игнорируется:
    берётся первое число, пропись нужна для сверки, а не для расчёта.
    """
    # Число берём до скобки, а валюту ищем во всей строке: в «50 000 000 (пятьдесят
    # миллионов) тенге» она стоит уже после прописи.
    head = text.split("(", 1)[0]
    resolved = detect_currency(text) or currency
    if resolved is None:
        raise MoneyParseError(f"Валюта не указана ни в {text!r}, ни в вызове")

    match = _AMOUNT.search(head)
    if match is None:
        raise MoneyParseError(f"Не нашёл числа в {text!r}")

    whole, frac = _split_decimal(match.group("digits"))
    minor = int(whole) * MINOR_UNITS[resolved] + int(frac)

    # Множитель ищем только до следующего числа, иначе «500 000 тенге по счёту
    # 20 млн» умножится на миллион из совершенно другой части фразы.
    tail = head[match.end() :]
    next_number = re.search(r"\d", tail)
    window = tail[: next_number.start()] if next_number else tail

    # Дробная часть с множителем неоднозначности не создаёт: разрядный разделитель
    # отделяет ровно три цифры, десятичный — одну-две, так что «3,75 млн» читается
    # единственным образом. Умножение уже переведённых минорных единиц точное.
    if (mult := _MULTIPLIER.search(window)) is not None:
        minor *= _MULTIPLIERS[mult.group(0).lower()]

    return Money(currency=resolved, minor=-minor if match.group("sign") else minor)


def convert(amount: Money, rate: Decimal, to: Currency, mode: Rounding = Rounding.HALF_UP) -> Money:
    """Пересчёт по курсу с явным округлением до минорной единицы.

    Курс приходит из документа, а не из справочника, поэтому округление делается
    один раз здесь: пересчитывать уже пересчитанное — верный способ разойтись с
    ответом на копейку.
    """
    converted = quantize(amount.to_decimal() * rate, scale=2, mode=mode)
    return Money.from_decimal(converted, to)


def quantize(value: Decimal, scale: int, mode: Rounding) -> Decimal:
    """Округление всегда явное: режим и число знаков приходят из IR ковенанта."""
    return value.quantize(Decimal(1).scaleb(-scale), rounding=mode.decimal_mode)
