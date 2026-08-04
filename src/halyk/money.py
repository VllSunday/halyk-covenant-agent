"""Деньги внутри пайплайна — целое число тиынов. Обоснование в docs/adr/0002."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum

TIYN_IN_TENGE = 100

# Множители из текста договоров: «не менее 50 млн тенге», «оборот в тыс. тенге».
_MULTIPLIERS: dict[str, int] = {
    "тыс": 1_000,
    "тысяч": 1_000,
    "млн": 1_000_000,
    "миллион": 1_000_000,
    "млрд": 1_000_000_000,
    "миллиард": 1_000_000_000,
}

_NUMBER = re.compile(
    r"(?P<sign>[-−–])?\s*"
    r"(?P<int>\d{1,3}(?:[\s  ']\d{3})+|\d+)"
    r"(?:[.,](?P<frac>\d{1,2}))?"
)
_MULTIPLIER = re.compile("|".join(sorted(_MULTIPLIERS, key=len, reverse=True)), re.IGNORECASE)


class Rounding(StrEnum):
    HALF_UP = "half_up"
    HALF_EVEN = "half_even"

    @property
    def decimal_mode(self) -> str:
        return ROUND_HALF_UP if self is Rounding.HALF_UP else ROUND_HALF_EVEN


class MoneyParseError(ValueError):
    pass


def parse_amount(text: str) -> int:
    """Разбирает денежную величину из текста документа в тиыны.

    Понимает разделители разрядов (пробел, неразрывный пробел, апостроф), запятую и
    точку как десятичный разделитель, а также словесные множители. Скобочная запись
    прописью — «50 000 000 (пятьдесят миллионов)» — игнорируется: берётся первое
    число, прописи в скобках нужны для сверки, а не для расчёта.
    """
    head = text.split("(", 1)[0]
    match = _NUMBER.search(head)
    if match is None:
        raise MoneyParseError(f"Не нашёл числа в {text!r}")

    digits = re.sub(r"[\s  ']", "", match.group("int"))
    frac = (match.group("frac") or "").ljust(2, "0")
    tiyn = int(digits) * TIYN_IN_TENGE + int(frac)

    # Множитель ищем только до следующего числа, иначе «500 000 тенге по счёту
    # 20 млн» умножится на миллион из совершенно другой части фразы.
    tail = head[match.end() :]
    next_number = re.search(r"\d", tail)
    window = tail[: next_number.start()] if next_number else tail

    if (mult := _MULTIPLIER.search(window)) is not None:
        if match.group("frac"):
            raise MoneyParseError(
                f"Дробная часть вместе с множителем в {text!r} — формулировка неоднозначная"
            )
        tiyn *= _MULTIPLIERS[mult.group(0).lower()]

    return -tiyn if match.group("sign") else tiyn


def tenge_to_tiyn(value: Decimal | int | str) -> int:
    try:
        exact = Decimal(value) * TIYN_IN_TENGE
    except InvalidOperation as exc:
        raise MoneyParseError(f"Не привёл {value!r} к сумме") from exc
    if exact != exact.to_integral_value():
        raise MoneyParseError(f"{value!r} содержит доли тиына")
    return int(exact)


def tiyn_to_tenge(tiyn: int) -> Decimal:
    """Обратное преобразование для сборки ответа. Возвращает точное значение."""
    return Decimal(tiyn) / TIYN_IN_TENGE


def quantize(value: Decimal, scale: int, mode: Rounding) -> Decimal:
    """Округление всегда явное: режим и число знаков приходят из IR ковенанта."""
    return value.quantize(Decimal(1).scaleb(-scale), rounding=mode.decimal_mode)
