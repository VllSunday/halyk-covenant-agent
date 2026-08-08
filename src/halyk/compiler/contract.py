"""Контракт компилятора: что модель обязана вернуть на один пункт договора.

Модель здесь не считает и не решает — она переводит текст пункта в исполнимое
дерево и называет, каких величин этому дереву не хватает. Всё остальное делает код,
поэтому контракт устроен так, чтобы каждое утверждение модели можно было проверить
по документу, не спрашивая её снова.

Отсюда три обязательных поля рядом с формулой: цитата, адрес страницы и перечень
требуемых фактов. Цитата проверяется поиском по исходному тексту, адрес — сверкой
с переписью, перечень фактов — сопоставлением с тем, что нашёл детерминированный
слой. Ответ, который нельзя проверить, к расчёту не допускается.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from halyk.models.covenant import Unit
from halyk.models.fact import Scope
from halyk.models.formula import CovenantFormula
from halyk.money import Currency, Money


class Resolution(StrEnum):
    """Судьба требуемой величины после сверки с детерминированным слоем."""

    RESOLVED = "resolved"
    NEEDS_RESOLUTION = "needs_resolution"
    # Требованию удовлетворяет больше одного факта. Выбор «первого попавшегося» дал
    # бы результат, зависящий от порядка чтения документов, поэтому это отказ.
    AMBIGUOUS = "ambiguous"


class Cardinality(StrEnum):
    """Сколько фактов удовлетворяет требованию."""

    ONE = "one"
    MANY = "many"


class FactRequirement(BaseModel):
    """Величина, без которой формулу не посчитать.

    Объявляется компилятором, находится отдельно. Совпадения по названию вида мало:
    порог существенности прошлого года, сумма в другой валюте и обязательство
    дочерней компании — разные величины с одним `fact_kind`. Поэтому требование
    несёт период, единицу, валюту и область, и сопоставление идёт по ним всем.
    """

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    scenario_id: str
    account_id: str

    fact_kind: str
    description: str
    unit: Unit
    currency: Currency | None = None
    scope: Scope = Scope.BORROWER
    cardinality: Cardinality = Cardinality.ONE

    period_start: date
    period_end: date

    # Уточнения отбора внутри вида: контрагент и порог существенности. У разовых
    # статей второе решает, попадёт ли статья в EBITDA вообще.
    counterparty: str | None = None
    materiality_floor: Money | None = None

    resolution: Resolution = Resolution.NEEDS_RESOLUTION

    @property
    def is_open(self) -> bool:
        return self.resolution is not Resolution.RESOLVED

    def matches_period(self, start: date, end: date) -> bool:
        return (self.period_start, self.period_end) == (start, end)


class CompiledClause(BaseModel):
    """Один пункт, переведённый в исполнимую форму."""

    model_config = ConfigDict(frozen=True)

    formula: CovenantFormula
    required_facts: tuple[FactRequirement, ...] = ()

    # Период измерения приходит из договора, но проверяется снаружи: он общий для
    # всех пунктов заёмщика, и расхождение между пунктами означает ошибку чтения,
    # а не разные периоды. Обоснование в docs/adr/0008.
    period_start: date
    period_end: date

    # То, что модель прочитала, но перевести в дерево не смогла. Пустой список здесь
    # ценнее длинного: он означает, что пункт разобран целиком.
    unresolved_terms: tuple[str, ...] = ()

    @property
    def cell(self) -> tuple[str, str]:
        return self.formula.scenario_id, self.formula.clause_id

    @property
    def open_requirements(self) -> tuple[FactRequirement, ...]:
        return tuple(item for item in self.required_facts if item.is_open)

    def requirement(self, requirement_id: str) -> FactRequirement | None:
        return next(
            (item for item in self.required_facts if item.requirement_id == requirement_id), None
        )


class CompilerResponse(BaseModel):
    """Ответ модели на одного заёмщика: все пункты его действующего договора."""

    model_config = ConfigDict(frozen=True)

    clauses: tuple[CompiledClause, ...] = Field(min_length=1)


# Отпечаток контракта идёт в ключ кэша: изменив форму ответа, мы обязаны получить
# промах, а не старую запись, разобранную по новым правилам.
OUTPUT_CONTRACT = "covenant-formula-v2"

INSTRUCTIONS = """
Ты переводишь финансовые ковенанты кредитного договора в исполнимые выражения. Ты
ничего не считаешь и не выносишь вердикт: числа возьмёт из реестра детерминированный
исполнитель.

Правила:

1. Верни отдельный пункт на каждый финансовый ковенант статьи, с номером ровно в том
   виде, в каком он напечатан в договоре.
2. Дерево собирается только из перечисленных операций. Если выражение не собирается,
   не подменяй его похожим — назови недостающее в unresolved_terms.
3. measure — величина, которую ограничивает пункт, и именно она пойдёт в ответ. Для
   условных пунктов она считается всегда, даже когда условие не сработало.
4. Каждую величину, которой нет в реестре операций, объяви в required_facts вместе с
   периодом, единицей, валютой и областью: показатель заёмщика и показатель группы —
   разные величины, даже когда называются одинаково.
5. quote — дословный фрагмент пункта, по которому видно порог и оператор. Он будет
   проверен поиском по тексту документа, поэтому не пересказывай его.
6. source_refs — страница, на которой этот фрагмент напечатан.
7. confidence снижай, когда формулировка допускает второе прочтение.
"""
