"""Проверка размерностей дерева до расчёта.

Дерево, собранное моделью, бывает формально валидным и бессмысленным: сумма денег с
отношением проходит и схему, и все структурные проверки, а исполнитель на ней
досчитает до конца и выдаст число. Число будет неправильным, а вердикт —
уверенным.

Проверка статическая: она смотрит только на форму дерева и на объявленные единицы,
не трогая ни реестра, ни документов. Поэтому её можно гонять до любого чтения
данных, и её отказ означает ошибку компиляции, а не свойство набора.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from halyk.compiler.contract import CompiledClause, FactRequirement
from halyk.models.covenant import Unit
from halyk.models.formula import (
    Absolute,
    Constant,
    Difference,
    ExternalMetric,
    FactValue,
    Largest,
    LedgerCount,
    LedgerSum,
    Product,
    Quantity,
    Ratio,
    Smallest,
    Sum,
)


class Dimension(StrEnum):
    MONEY = "money"
    RATIO = "ratio"
    COUNT = "count"
    DAYS = "days"
    # Константа приобретает размерность соседа: `EBITDA × 1.5` и `выручка ≥ 0.8`
    # одинаково законны, и требовать от модели объявлять её единицу значило бы
    # плодить отказы там, где смысл однозначен.
    SCALAR = "scalar"


_FROM_UNIT: dict[Unit, Dimension] = {
    Unit.MONEY: Dimension.MONEY,
    Unit.RATIO: Dimension.RATIO,
    Unit.PERCENT: Dimension.RATIO,
    Unit.COUNT: Dimension.COUNT,
    Unit.DAYS: Dimension.DAYS,
}


class DimensionError(ValueError):
    """Размерности узла не сходятся. Считать такое дерево нельзя."""

    def __init__(self, path: str, detail: str) -> None:
        super().__init__(f"{detail} (узел {path})")
        self.path = path
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Typed:
    dimension: Dimension
    path: str


def _requirements(clause: CompiledClause) -> dict[str, FactRequirement]:
    return {item.fact_kind: item for item in clause.required_facts}


def _unify(left: Dimension, right: Dimension) -> Dimension | None:
    """Общая размерность двух величин при сложении и сравнении."""
    if left is right:
        return left
    if Dimension.SCALAR in (left, right):
        return right if left is Dimension.SCALAR else left
    return None


def infer(node: Quantity, known: dict[str, FactRequirement], path: str = "measure") -> Typed:
    """Размерность узла или отказ с адресом.

    Правила выведены из смысла величин, а не из удобства: деньги на деньги дают
    отношение, деньги на отношение — снова деньги, а количество с деньгами не
    складывается ни при каких обстоятельствах.
    """
    return Typed(_dimension_of(node, known, path), path)


def _dimension_of(node: Quantity, known: dict[str, FactRequirement], path: str) -> Dimension:
    """Разбор по типу узла таблицей, а не ветвлениями.

    Соответствие «узел — правило» здесь именно таблица: добавив узел в дерево и
    забыв про его размерность, разработчик получит отказ на первом же расчёте, а не
    молча выведенную не ту единицу.
    """
    if (leaf := _LEAF_DIMENSIONS.get(type(node))) is not None:
        return leaf
    return _COMPOSITE_RULES[type(node)](node, known, path)


def _difference(node: Difference, known: dict[str, FactRequirement], path: str) -> Dimension:
    left = infer(node.left, known, f"{path}.left")
    right = infer(node.right, known, f"{path}.right")
    if (common := _unify(left.dimension, right.dimension)) is None:
        raise DimensionError(path, f"вычитание {right.dimension} из {left.dimension} не определено")
    return common


def _declared(
    node: FactValue | ExternalMetric, known: dict[str, FactRequirement], path: str
) -> Dimension:
    """Размерность факта берётся из объявленного требования, а не угадывается."""
    name = node.fact_kind if isinstance(node, FactValue) else node.name
    requirement = known.get(name)
    if requirement is None:
        raise DimensionError(path, f"величина {name} не объявлена в required_facts")
    return _FROM_UNIT[requirement.unit]


def _parts(
    node: Sum | Largest | Smallest, known: dict[str, FactRequirement], path: str
) -> list[Typed]:
    children = node.terms if isinstance(node, Sum) else node.values
    label = "terms" if isinstance(node, Sum) else "values"
    return [infer(child, known, f"{path}.{label}[{i}]") for i, child in enumerate(children)]


def _homogeneous(parts: list[Typed], path: str, verb: str) -> Dimension:
    common = parts[0].dimension
    for part in parts[1:]:
        if (merged := _unify(common, part.dimension)) is None:
            raise DimensionError(path, f"{verb} {common} и {part.dimension} нельзя")
        common = merged
    return common


def _product(node: Product, known: dict[str, FactRequirement], path: str) -> Dimension:
    """Умножать на безразмерный множитель можно, размерности между собой — нет."""
    parts = [infer(f, known, f"{path}.factors[{i}]") for i, f in enumerate(node.factors)]
    dimensional = [part for part in parts if part.dimension is not Dimension.SCALAR]
    if len(dimensional) > 1:
        names = ", ".join(str(part.dimension) for part in dimensional)
        raise DimensionError(path, f"произведение размерных величин не определено: {names}")
    return dimensional[0].dimension if dimensional else Dimension.SCALAR


def _ratio(node: Ratio, known: dict[str, FactRequirement], path: str) -> Dimension:
    numerator = infer(node.numerator, known, f"{path}.numerator")
    denominator = infer(node.denominator, known, f"{path}.denominator")
    top, bottom = numerator.dimension, denominator.dimension

    if bottom is Dimension.SCALAR:
        return top
    if top is bottom:
        # Однородное отношение безразмерно: доля выручки, покрытие процентов.
        return Dimension.RATIO
    if bottom is Dimension.RATIO:
        return top
    raise DimensionError(path, f"отношение {top} к {bottom} не определено")


def check(clause: CompiledClause) -> list[DimensionError]:
    """Размерности всего пункта: величина, порог и условие срабатывания."""
    known = _requirements(clause)
    found: list[DimensionError] = []

    measure: Dimension | None = None
    try:
        measure = infer(clause.formula.measure, known).dimension
    except DimensionError as exc:
        found.append(exc)

    try:
        threshold = infer(clause.formula.threshold, known, "threshold").dimension
    except DimensionError as exc:
        found.append(exc)
    else:
        if measure is not None and _unify(measure, threshold) is None:
            found.append(
                DimensionError("threshold", f"порог {threshold} несравним с величиной {measure}")
            )

    # Объявленная единица ячейки должна совпадать с тем, что получилось из дерева:
    # `actual` уходит в ответ именно в ней.
    declared = _FROM_UNIT[clause.formula.unit]
    if measure is not None and _unify(measure, declared) is None:
        found.append(DimensionError("unit", f"объявлена единица {declared}, дерево даёт {measure}"))

    if (condition := clause.formula.applies_when) is not None:
        try:
            left = infer(condition.left, known, "applies_when.left").dimension
            right = infer(condition.right, known, "applies_when.right").dimension
        except DimensionError as exc:
            found.append(exc)
        else:
            if _unify(left, right) is None:
                found.append(DimensionError("applies_when", f"условие сравнивает {left} с {right}"))
    return found


_LEAF_DIMENSIONS: dict[type, Dimension] = {
    LedgerSum: Dimension.MONEY,
    LedgerCount: Dimension.COUNT,
    Constant: Dimension.SCALAR,
}

_COMPOSITE_RULES: dict[type, Callable[[Any, dict[str, FactRequirement], str], Dimension]] = {
    FactValue: _declared,
    ExternalMetric: _declared,
    Sum: lambda node, known, path: _homogeneous(_parts(node, known, path), path, "складывать"),
    Largest: lambda node, known, path: _homogeneous(_parts(node, known, path), path, "сравнивать"),
    Smallest: lambda node, known, path: _homogeneous(_parts(node, known, path), path, "сравнивать"),
    Difference: _difference,
    Product: _product,
    Ratio: _ratio,
    Absolute: lambda node, known, path: infer(node.value, known, f"{path}.value").dimension,
}
