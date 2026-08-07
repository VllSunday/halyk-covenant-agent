"""Внешний контракт ответа. Форма задана организаторами, а не нашей моделью."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, WithJsonSchema

from halyk.models.result import Verdict
from halyk.money import Rounding, quantize

ACTUAL_SCALE = 2


def _round_actual(value: Decimal) -> float:
    """Округление до двух знаков — последнее, что происходит с числом.

    Вердикт считается по сырому значению: округлив раньше, можно перевернуть
    сравнение с порогом на границе. `float` здесь потому, что ключ хранит число, а
    не строку, и погрешность double на двух знаках заведомо ниже допуска скоринга.
    """
    return float(quantize(value, ACTUAL_SCALE, Rounding.HALF_UP))


ActualValue = Annotated[
    Decimal,
    Field(ge=0),
    PlainSerializer(_round_actual, return_type=float, when_used="always"),
    # Свой сериализатор подменяет схему на голое `number` и роняет границу из
    # Field(ge=0). Без этой строки `halyk validate` без шаблона принял бы
    # отрицательное значение, которое модель в рантайме отвергает.
    WithJsonSchema({"type": "number", "minimum": 0}, mode="serialization"),
]


class CovenantCell(BaseModel):
    """Ячейка ответа: вердикт, число и улика.

    `actual` неотрицателен по условию задачи — в реестре расходы записаны со знаком
    минус, и потерянный `abs` иначе уезжает прямо в ответ.

    `evidence_txn_id` обязателен и при пустой улике: `null` — это значение, а не
    основание выкинуть ключ из ячейки.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Verdict
    actual: ActualValue
    evidence_txn_id: str | None


class Submission(BaseModel):
    """Файл ответа целиком.

    Ключи `answers` не перечислены в модели: их состав диктует шаблон организаторов,
    и сверяется он с ним же, а не с нашим представлением о наборе заёмщиков.

    Посторонние поля запрещены: это внешний контракт, и молча проглоченный лишний
    ключ означал бы, что мы отправили не тот документ, который проверяли.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    team: str
    contact_email: str
    model: str
    answers: dict[str, dict[str, CovenantCell]]

    def cell_keys(self) -> set[tuple[str, str]]:
        return {
            (scenario, covenant)
            for scenario, covenants in self.answers.items()
            for covenant in covenants
        }
