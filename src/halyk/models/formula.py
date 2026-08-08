"""Formula AST — исполнимая форма одного ковенанта.

Плоская спецификация (`CovenantIR`) описывала одну агрегацию с фильтрами и порогом.
Реальные пункты так не устроены: покрытие процентов делит одну вычисленную величину
на другую, EBITDA сама собирается из выручки, расходов и разовых статей из
примечаний, а «оплата труда и коммунальные услуги — по отдельности, а не в
совокупности» сравнивает с потолком максимум из двух независимых сумм. Всё это
выражается деревом, но не парой «агрегация + фильтр».

Список узлов закрыт намеренно: исполнитель обязан уметь посчитать любое дерево,
прошедшее валидацию, без обращения к модели и без интерпретации свободного текста.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from halyk.models.classification import TransactionCategory
from halyk.models.covenant import Operator, RoundingSpec, Unit
from halyk.models.source import SourceRef


class Direction(StrEnum):
    """Знак операции в реестре. Расходы записаны отрицательными суммами."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"
    ANY = "any"


class RelatedParty(StrEnum):
    """Отбор по связанности контрагента.

    Порог связанности свой у каждого заёмщика и объявлен в его досье KYC, поэтому
    здесь стоит только намерение — конкретные контрагенты подставляются исполнителем
    из фактов.
    """

    ANY = "any"
    ONLY = "only"
    EXCLUDE = "exclude"


class Selector(BaseModel):
    """Какие строки реестра попадают в величину.

    Пустое поле означает «без ограничения по этому признаку», а не «ничего»: иначе
    любой селектор пришлось бы заполнять целиком, и ошибка компилятора выглядела бы
    как осмысленный фильтр.
    """

    model_config = ConfigDict(frozen=True)

    categories: tuple[TransactionCategory, ...] = ()
    direction: Direction = Direction.ANY
    counterparties: tuple[str, ...] = ()
    related_party: RelatedParty = RelatedParty.ANY
    # Календарные месяцы внутри периода: квартальные пункты («выручка за четвёртый
    # квартал») режут период, а не меняют его границы.
    months: tuple[int, ...] = Field(default=(), max_length=12)
    exclude_txn_ids: tuple[str, ...] = ()

    def is_open(self) -> bool:
        """Селектор без единого ограничения — почти всегда ошибка компиляции."""
        return not (
            self.categories
            or self.counterparties
            or self.months
            or self.exclude_txn_ids
            or self.direction is not Direction.ANY
            or self.related_party is not RelatedParty.ANY
        )


class LedgerSum(BaseModel):
    """Сумма модулей отобранных операций.

    Модуль, а не знак: расходы в реестре отрицательны, а `actual` по условию задачи
    всегда положителен. Знак несёт `Direction`, и складывать разнонаправленные строки
    без него бессмысленно.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["ledger_sum"] = "ledger_sum"
    selector: Selector


class LedgerCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["ledger_count"] = "ledger_count"
    selector: Selector


class FactValue(BaseModel):
    """Величина из документа, а не из реестра.

    Обязательство по программе выходных пособий, порог существенности разовых статей,
    доля дочерних организаций в залоге — всё это раскрыто только в примечаниях и
    отдельной проводки не имеет.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["fact_value"] = "fact_value"
    fact_kind: str
    # Отбор внутри вида факта: у заёмщика может быть несколько разовых статей, и
    # в формулу входят не все, а прошедшие порог существенности.
    description_contains: str | None = None
    counterparty: str | None = None
    above_one_off_policy: bool = False


class Constant(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["constant"] = "constant"
    value: Decimal


class ExternalMetric(BaseModel):
    """Величина, которой в датасете может не быть.

    Существует ради честного отказа: подставить сюда число, восстановленное из
    эталона, значит получить правильный ответ на открытом наборе и молчаливо неверный
    на приватном. Исполнитель обязан вернуть отказ, а не догадку.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["external"] = "external"
    name: str
    description: str


class Sum(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["add"] = "add"
    terms: tuple[Quantity, ...] = Field(min_length=2)


class Difference(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["sub"] = "sub"
    left: Quantity
    right: Quantity


class Product(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["mul"] = "mul"
    factors: tuple[Quantity, ...] = Field(min_length=2)


class Ratio(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["div"] = "div"
    numerator: Quantity
    denominator: Quantity


class Absolute(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["abs"] = "abs"
    value: Quantity


class Largest(BaseModel):
    """Максимум из нескольких величин.

    Так выражается «по отдельности, а не в совокупности»: потолок нарушен, если его
    перешла хотя бы одна из статей, и в `actual` идёт та, что ближе всего к порогу.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["max"] = "max"
    values: tuple[Quantity, ...] = Field(min_length=2)


class Smallest(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["min"] = "min"
    values: tuple[Quantity, ...] = Field(min_length=2)


Quantity = Annotated[
    LedgerSum
    | LedgerCount
    | FactValue
    | Constant
    | ExternalMetric
    | Sum
    | Difference
    | Product
    | Ratio
    | Absolute
    | Largest
    | Smallest,
    Field(discriminator="op"),
]

for _node in (Sum, Difference, Product, Ratio, Absolute, Largest, Smallest):
    _node.model_rebuild()


class Condition(BaseModel):
    """Сравнение двух величин. Само по себе вердикта не даёт."""

    model_config = ConfigDict(frozen=True)

    left: Quantity
    operator: Operator
    right: Quantity


class EvidenceMode(StrEnum):
    """Как выбирается транзакция-улика.

    Контрфактически — значит перебором: строка считается уликой, если без неё вердикт
    меняется. Правил вида «самая крупная» здесь нет намеренно, условие задачи прямо
    отвергает их.
    """

    COUNTERFACTUAL = "counterfactual"
    NONE = "none"


class CovenantFormula(BaseModel):
    """Полная спецификация ковенанта: что считаем, с чем сравниваем, когда применяем."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    clause_id: str
    title: str

    # То самое число, которое уходит в `actual`. Для условных пунктов оно считается
    # всегда, даже когда пункт не применяется: условие задачи требует фактическое
    # значение показателя, а не значение условия.
    measure: Quantity
    operator: Operator
    threshold: Quantity
    unit: Unit

    # Пункт со срабатыванием: пока условие не выполнено, нарушения быть не может.
    applies_when: Condition | None = None
    rounding: RoundingSpec = RoundingSpec()
    evidence: EvidenceMode = EvidenceMode.COUNTERFACTUAL

    source_refs: tuple[SourceRef, ...] = Field(min_length=1)
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)
