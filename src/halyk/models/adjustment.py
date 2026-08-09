"""Корректировки реестра: что документ велит изменить в операции и применилось ли это.

Исходная строка реестра неизменна. Корректировка описывает изменение отдельно от
данных, поэтому в объяснении ответа всегда видно, какое число пришло из выгрузки, а
какое — из документа, и по чьему указанию.

Тип корректировки задаёт её обязательные поля: у переклассификации это новая
категория, у пересчёта — курс и обе валюты. Общая модель со всеми полями
необязательными позволяла собрать корректировку без предмета и потом отчитаться,
что она применена, — поэтому здесь размеченное объединение, а не один класс на все
случаи.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from halyk.hashing import sha256_payload
from halyk.models.source import SourceRef
from halyk.models.transaction import LedgerRow
from halyk.money import Currency, CurrencyMismatchError, Money


class AdjustmentAction(StrEnum):
    RECLASSIFY = "reclassify"
    SET_MISSING_AMOUNT = "set_missing_amount"
    EXCLUDE = "exclude"
    INCLUDE = "include"
    SET_EFFECTIVE_DATE = "set_effective_date"
    CONVERT_CURRENCY = "convert_currency"
    REVIEW_NO_CHANGE = "review_no_change"


class AdjustmentStatus(StrEnum):
    """Судьба корректировки.

    Отклонённое и неприменённое хранится наравне с применённым: половина ловушек
    датасета в том, что документ рассматривает вопрос и оставляет всё как было, и
    отличить «не нашли» от «нашли и отказали» нужно в отчёте, а не в голове.
    """

    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    UNCONFIRMED = "unconfirmed"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INEFFECTIVE = "ineffective"
    CORROBORATED = "corroborated"

    @property
    def is_problem(self) -> bool:
        """Итог, при котором документ остался неучтённым по нашей вине.

        Отклонённое самим документом сюда не входит: там всё сработало как надо.
        """
        return self in (
            AdjustmentStatus.UNMATCHED,
            AdjustmentStatus.AMBIGUOUS,
            AdjustmentStatus.CONFLICTING,
            AdjustmentStatus.INEFFECTIVE,
            AdjustmentStatus.PENDING,
        )


class SelectorError(ValueError):
    pass


class TransactionSelector(BaseModel):
    """Как документ указывает на операцию.

    Идентификатор есть не всегда: окончательный отчёт аудитора называет сумму и
    контрагента, но не номер проводки. Поэтому отбор описан явно и допускает оба
    способа, а результат сопоставления фиксируется отдельным статусом.
    """

    model_config = ConfigDict(frozen=True)

    txn_id: str | None = None
    counterparty: str | None = None
    amount: Money | None = None

    @model_validator(mode="after")
    def _at_least_one_criterion(self) -> Self:
        if self.txn_id is None and self.counterparty is None and self.amount is None:
            raise SelectorError("Пустой отбор совпал бы с любой операцией реестра")
        return self

    def describe(self) -> str:
        parts = [
            self.txn_id,
            self.counterparty,
            str(self.amount.to_decimal()) if self.amount else None,
        ]
        return " / ".join(part for part in parts if part)


class AdjustmentBase(BaseModel):
    """Общая часть: к кому относится, откуда взято и чем закончилось."""

    model_config = ConfigDict(frozen=True)

    account_id: str
    selector: TransactionSelector
    reason: str
    source: SourceRef
    status: AdjustmentStatus = AdjustmentStatus.PENDING
    txn_id: str | None = None
    note: str = ""

    # Поля нормализованной операции, которые меняет это действие.
    changed_fields: ClassVar[frozenset[str]] = frozenset()

    # Не влияет на смысл корректировки: это её судьба, а не содержание.
    _VOLATILE: ClassVar[frozenset[str]] = frozenset(
        {"status", "txn_id", "note", "reason", "source"}
    )

    @property
    def id(self) -> str:
        """Идентификатор по всему, что влияет на результат применения.

        Курс, сумма и дата входят в него наравне с отбором: две корректировки одной
        операции с разными курсами — разные корректировки, и в lineage они не должны
        сливаться в одну запись.
        """
        payload = self.model_dump(mode="json", exclude=set(self._VOLATILE))
        payload["source_hash"] = self.source.file_hash
        payload["source_page"] = self.source.page
        return "adj-" + sha256_payload(payload)[:16]

    def effect(self) -> dict[str, Any]:
        """Что именно корректировка запишет в операцию.

        По этим значениям, а не по одним именам полей, различаются спор и
        подтверждение: два документа, называющие ту же новую категорию, друг другу
        не противоречат, и объявлять их конфликтом значит терять оба.
        """
        return {}

    def resolved(self, status: AdjustmentStatus, txn_id: str | None = None, note: str = "") -> Self:
        return self.model_copy(update={"status": status, "txn_id": txn_id, "note": note})

    def record(self) -> dict[str, Any]:
        return {"id": self.id} | self.model_dump(mode="json")


class ReclassifyAdjustment(AdjustmentBase):
    """Операция отнесена к другой статье для целей ковенантов."""

    action: Literal[AdjustmentAction.RECLASSIFY] = AdjustmentAction.RECLASSIFY
    new_value: str = Field(min_length=1)
    old_value: str | None = None

    changed_fields: ClassVar[frozenset[str]] = frozenset({"covenant_category"})

    def effect(self) -> dict[str, Any]:
        return {"covenant_category": self.new_value}


class SetMissingAmountAdjustment(AdjustmentBase):
    """Сумма, не попавшая в выгрузку, раскрыта документом.

    Именно не попавшая: заполнять можно только пустое место. Строка с суммой —
    свидетельство, и подменять её числом из примечаний нельзя, даже если числа
    расходятся; такое расхождение разбирают, а не затирают.
    """

    action: Literal[AdjustmentAction.SET_MISSING_AMOUNT] = AdjustmentAction.SET_MISSING_AMOUNT
    amount: Money

    changed_fields: ClassVar[frozenset[str]] = frozenset({"amount"})

    def effect(self) -> dict[str, Any]:
        return {"amount": str(self.amount.to_decimal()) + " " + self.amount.currency}


class ExcludeAdjustment(AdjustmentBase):
    """Операция не относится к ковенантному периоду."""

    action: Literal[AdjustmentAction.EXCLUDE] = AdjustmentAction.EXCLUDE

    changed_fields: ClassVar[frozenset[str]] = frozenset({"in_period"})

    def effect(self) -> dict[str, Any]:
        return {"in_period": False}


class IncludeAdjustment(AdjustmentBase):
    """Операция относится к ковенантному периоду, хотя её дата лежит вне него.

    Дата при этом не подменяется: документ говорит о принадлежности периоду, а не о
    том, каким днём операция должна была быть проведена. Подставить сюда край периода
    значило бы придумать день и сдвинуть операцию в квартальных выборках.
    """

    action: Literal[AdjustmentAction.INCLUDE] = AdjustmentAction.INCLUDE

    changed_fields: ClassVar[frozenset[str]] = frozenset({"in_period"})

    def effect(self) -> dict[str, Any]:
        return {"in_period": True}


class SetEffectiveDateAdjustment(AdjustmentBase):
    """Период определяется датой оказания услуг, а не датой счёта-фактуры."""

    action: Literal[AdjustmentAction.SET_EFFECTIVE_DATE] = AdjustmentAction.SET_EFFECTIVE_DATE
    effective_date: date

    changed_fields: ClassVar[frozenset[str]] = frozenset({"effective_date"})

    def effect(self) -> dict[str, Any]:
        return {"effective_date": self.effective_date.isoformat()}


class ConvertCurrencyAdjustment(AdjustmentBase):
    """Пересчёт по курсу, раскрытому в примечаниях."""

    action: Literal[AdjustmentAction.CONVERT_CURRENCY] = AdjustmentAction.CONVERT_CURRENCY
    rate: Decimal = Field(gt=0)
    from_currency: Currency
    to_currency: Currency

    changed_fields: ClassVar[frozenset[str]] = frozenset({"amount"})

    @model_validator(mode="after")
    def _currencies_differ(self) -> Self:
        # Пересчёт валюты в неё же — это умножение суммы на число, и никакой
        # документ такого не предписывал.
        if self.from_currency is self.to_currency:
            raise CurrencyMismatchError(
                f"Пересчёт {self.from_currency} в {self.to_currency} меняет сумму, а не валюту"
            )
        return self

    def effect(self) -> dict[str, Any]:
        return {"amount": f"{self.from_currency}->{self.to_currency}@{self.rate}"}


class ReviewNoChangeAdjustment(AdjustmentBase):
    """Вопрос рассмотрен, первоначальная классификация сохранена.

    Изменения здесь нет по определению, и всё же это запись в том же потоке:
    датасет намеренно подсовывает рассмотренные и отклонённые переклассификации, и
    отличить «мы не заметили» от «документ отказал» можно только по такой отметке.
    """

    action: Literal[AdjustmentAction.REVIEW_NO_CHANGE] = AdjustmentAction.REVIEW_NO_CHANGE
    considered: str | None = None
    old_value: str | None = None
    status: AdjustmentStatus = AdjustmentStatus.REJECTED

    # Судьба задана контрактом: у записи нет предмета, и «применить» её нельзя ни
    # из документа, ни по ошибке в коде. Заменённая и неподтверждённая ведомость
    # остаются допустимыми — там отказ пришёл не от самого документа.
    _ALLOWED: ClassVar[frozenset[AdjustmentStatus]] = frozenset(
        {
            AdjustmentStatus.REJECTED,
            AdjustmentStatus.SUPERSEDED,
            AdjustmentStatus.UNCONFIRMED,
        }
    )

    @model_validator(mode="after")
    def _never_applied(self) -> Self:
        if self.status not in self._ALLOWED:
            raise ValueError(
                f"Рассмотрение без изменений не может получить статус {self.status.value}"
            )
        return self


Adjustment = Annotated[
    ReclassifyAdjustment
    | SetMissingAmountAdjustment
    | ExcludeAdjustment
    | IncludeAdjustment
    | SetEffectiveDateAdjustment
    | ConvertCurrencyAdjustment
    | ReviewNoChangeAdjustment,
    Field(discriminator="action"),
]

AdjustmentAdapter: TypeAdapter[Adjustment] = TypeAdapter(Adjustment)


class NormalisedTransaction(BaseModel):
    """Операция в том виде, в котором её увидит расчёт.

    Рядом лежит исходная строка: сравнение «было — стало» не требует второго
    источника, а список применённых корректировок объясняет каждое расхождение.
    """

    model_config = ConfigDict(frozen=True)

    row: LedgerRow
    amount: Money | None
    effective_date: date
    in_period: bool = True
    covenant_category: str | None = None
    excluded: bool = False
    adjustments: tuple[str, ...] = ()
    # Состояние операции до первой применённой корректировки. Нужно для выбора улики:
    # по условию задачи ею считается операция, чья правка документом и приводит к
    # нарушению, а проверить это можно только откатив правку. Восстанавливать состояние
    # заново нельзя — часть его (статья, назначенная классификатором) к тому моменту
    # уже не выводится из исходной строки.
    before: NormalisedTransaction | None = None

    @property
    def txn_id(self) -> str:
        return self.row.txn_id

    @property
    def is_adjusted(self) -> bool:
        return bool(self.adjustments)

    def state(self) -> dict[str, Any]:
        """Значения, по которым проверяется, был ли у корректировки эффект.

        `excluded` сюда не входит: это служебная отметка, из которой вычисляется
        `in_period`. Операция, и без документа не попавшая в период, от исключения
        не меняется — и корректировка должна об этом сообщить, а не отчитаться
        об успехе. `before` — происхождение, а не значение: сам факт снимка не
        делает корректировку подействовавшей.
        """
        return self.model_dump(mode="json", exclude={"row", "adjustments", "excluded", "before"})

    def record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
