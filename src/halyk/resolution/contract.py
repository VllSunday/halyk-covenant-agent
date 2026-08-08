"""Контракт resolver: что модель возвращает на требования, которые остались открытыми.

Компилятор объявил, каких величин не хватает; детерминированный слой нашёл те, что
опознаются формулировками. Остальное дочитывает модель — и только дочитывает.

Отсюда три ограничения формы ответа.

Вердикта в нём нет. Ни статуса корректировки, ни ячейки Submission: судьбу
корректировки назначает код по праву документа менять расчёт, а ячейку собирает
исполнитель из дерева. Дай мы модели поле статуса — она заполнила бы его правдоподобно,
и отличить прочитанное от додуманного стало бы нечем.

Каждое утверждение адресуемо. Имя файла, номер страницы и дословная цитата — по ним
ответ проверяется поиском по тексту, без второго обращения к модели.

Выведенная величина возвращается слагаемыми, а не одним числом. Капитальные затраты в
отчётности отдельной строкой не стоят, они выводятся из движения основных средств; и
если модель называет только результат, проверить его нечем, а если слагаемые — их
складывает код, а расхождение с её собственным итогом означает, что верить нельзя ни
тому, ни другому.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from halyk.models.adjustment import AdjustmentAction
from halyk.models.covenant import Unit
from halyk.money import Currency

# Отпечаток контракта идёт в ключ кэша: изменив форму ответа, мы обязаны получить
# промах, а не старую запись, разобранную по новым правилам.
OUTPUT_CONTRACT = "requirement-resolution-v1"

SCHEMA_NAME = "resolver_response"


class Sign(StrEnum):
    """Знак слагаемого в выводимой величине."""

    PLUS = "plus"
    MINUS = "minus"

    @property
    def factor(self) -> int:
        return 1 if self is Sign.PLUS else -1


class Evidence(BaseModel):
    """Адрес утверждения: где напечатано и что именно там написано."""

    model_config = ConfigDict(frozen=True)

    file_name: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class ResolvedFact(BaseModel):
    """Величина, прочитанная под конкретное требование компилятора."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    fact_kind: str
    amount: Decimal
    unit: Unit
    currency: Currency | None = None
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class DerivationTerm(BaseModel):
    """Одно раскрытое число, входящее в выводимую величину."""

    model_config = ConfigDict(frozen=True)

    label: str
    sign: Sign
    amount: Decimal
    evidence: Evidence


class Derivation(BaseModel):
    """Величина, которой в отчётности нет строкой, но которая из неё выводится."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    fact_kind: str
    unit: Unit
    currency: Currency | None = None
    # Тождество словами: «closing = opening + additions − depreciation − disposals».
    # Нужно объяснению ответа, а не расчёту — считается всё равно по слагаемым.
    identity: str
    terms: tuple[DerivationTerm, ...] = Field(min_length=2)
    amount: Decimal
    confidence: float = Field(ge=0.0, le=1.0)

    def total(self) -> Decimal:
        """Сумма слагаемых со знаками. Именно она уходит в расчёт."""
        return sum((term.sign.factor * term.amount for term in self.terms), Decimal(0))


class ProposedAdjustment(BaseModel):
    """Изменение операции, предписанное документом.

    Поля обязательные для конкретного действия проверяются не здесь, а при сборке
    настоящей корректировки: там уже есть модель на каждое действие, и повторять её
    ограничения на проводе значило бы держать два описания одного контракта.
    """

    model_config = ConfigDict(frozen=True)

    action: AdjustmentAction
    reason: str = Field(min_length=1)
    evidence: Evidence

    txn_id: str | None = None
    counterparty: str | None = None
    selector_amount: Decimal | None = None
    selector_currency: Currency | None = None

    new_value: str | None = None
    old_value: str | None = None
    considered: str | None = None
    amount: Decimal | None = None
    currency: Currency | None = None
    effective_date: date | None = None
    rate: Decimal | None = None
    from_currency: Currency | None = None
    to_currency: Currency | None = None


class ResolverResponse(BaseModel):
    """Ответ модели на одного заёмщика.

    Все три списка пустые — законный ответ: он означает, что в документах заёмщика
    ничего под открытые требования не нашлось, и это честнее выдуманного числа.
    """

    model_config = ConfigDict(frozen=True)

    facts: tuple[ResolvedFact, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    adjustments: tuple[ProposedAdjustment, ...] = ()


INSTRUCTIONS = """
Ты дочитываешь величины, которых не хватило для расчёта ковенантов, по документам
одного заёмщика. Ты ничего не считаешь и не выносишь вердикт: считает и сравнивает
детерминированный исполнитель.

На вход приходит список открытых требований и тексты страниц. Отвечай только на те
требования, величину для которых видишь в тексте.

Правила:

1. Каждый элемент ответа ссылается на requirement_id из списка и повторяет его
   fact_kind. Требования, которого нет в списке, не выдумывай.
2. unit и currency обязаны совпадать с объявленными в требовании: величина заёмщика и
   величина группы, сумма в тенге и сумма в долларах — разные величины.
3. evidence — имя файла, номер страницы и дословный фрагмент, по которому видно
   число. Фрагмент будет проверен поиском по тексту страницы, поэтому не пересказывай
   его и не склеивай куски из разных мест.
4. Величину, которая напечатана строкой, возвращай в facts. Величину, которой строкой
   нет, но которая выводится из нескольких раскрытых чисел, — в derivations: назови
   слагаемые со знаками, каждое со своей цитатой, и запиши тождество словами.
5. Если требованию удовлетворяет больше одного числа, верни оба: выбирать будем не мы
   с тобой, а правило разрешения неоднозначности.
6. adjustments — то, что документ предписывает изменить в конкретной операции реестра.
   Статуса у корректировки нет: применилась она или нет, решает код.
7. Ничего не отвечай про требования, величины для которых в документах нет. Пустой
   ответ лучше правдоподобного числа.
"""
