"""Официальные исправления организаторов к текстам договоров.

Такое сообщение — источник более высокого приоритета, чем сам документ, и работает
оно по тем же правилам, что и окончательный отчёт аудитора против промежуточной
ведомости: перекрывает одно поле, названное явно, и ничего кроме.

Правку нельзя вписывать прямо в формулу. Условие «если заёмщик P4, то порог другой»
неотличимо от подгонки под публичный ключ: его не видно в артефакте, оно не имеет
основания и легко расползается на соседние ковенанты. Поэтому исправления объявлены
декларативно, применяются только при полном совпадении области, а исходное значение
остаётся в происхождении ответа рядом с применённым.
"""

from __future__ import annotations

import tomllib
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

from halyk.hashing import sha256_file
from halyk.models.covenant import CovenantIR
from halyk.models.formula import Constant, CovenantFormula

# Реестр лежит внутри пакета, а не рядом с рабочим каталогом: относительный путь
# зависел бы от того, откуда запущена команда, и при запуске из другого каталога
# исправление молча не применилось бы. Тихо посчитать по неверному порогу хуже,
# чем упасть.
RESOURCE = "official_errata.toml"


def _packaged_registry(stack: ExitStack) -> Path:
    source = resources.files("halyk.knowledge") / "data" / RESOURCE
    return stack.enter_context(resources.as_file(source))


class ErrataError(RuntimeError):
    pass


class Erratum(BaseModel):
    """Одно исправление: что, где и с какого на какое значение."""

    model_config = ConfigDict(frozen=True)

    id: str
    scenario: str
    covenant: str
    # Пока исправляются только пороги. Расширять список нужно вместе с кодом,
    # который умеет подставить соответствующее поле IR.
    field: Literal["threshold"]
    documented_value: Decimal
    corrected_value: Decimal
    announced_on: date
    reason: str
    source: str

    def covers(self, borrower_id: str, covenant_id: str) -> bool:
        return self.scenario == borrower_id and self.covenant == covenant_id


@dataclass(frozen=True, slots=True)
class ErratumApplication:
    """Отметка о применённом исправлении. Идёт в происхождение ответа."""

    erratum_id: str
    field: str
    documented_value: Decimal
    applied_value: Decimal
    reason: str
    source: str

    def record(self) -> dict[str, Any]:
        return {
            "erratum_id": self.erratum_id,
            "field": self.field,
            "documented_value": str(self.documented_value),
            "applied_value": str(self.applied_value),
            "reason": self.reason,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ErrataRegistry:
    """Реестр исправлений вместе с отпечатком файла, из которого он прочитан."""

    entries: tuple[Erratum, ...] = ()
    digest: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Прочитать реестр: по умолчанию тот, что поставляется с пакетом.

        Отсутствие файла — ошибка, а не пустой реестр. Молчаливо посчитать без
        официального исправления значит выдать неверное число с видом успешного
        прогона. Пустой реестр получается только явным `ErrataRegistry()`.
        """
        with ExitStack() as stack:
            source = path if path is not None else _packaged_registry(stack)
            if not source.is_file():
                raise ErrataError(f"Реестр официальных исправлений не найден: {source}")

            document: dict[str, Any] = tomllib.loads(source.read_text(encoding="utf-8"))
            entries = tuple(Erratum.model_validate(item) for item in document.get("erratum", ()))
            scopes = [(e.scenario, e.covenant, e.field) for e in entries]
            if len(set(scopes)) != len(scopes):
                raise ErrataError(f"В {source} два исправления на одно поле одного ковенанта")
            return cls(entries=entries, digest=sha256_file(source))

    def for_covenant(self, borrower_id: str, covenant_id: str) -> Erratum | None:
        return next((e for e in self.entries if e.covers(borrower_id, covenant_id)), None)

    def apply(self, ir: CovenantIR) -> tuple[CovenantIR, ErratumApplication | None]:
        """Наложить исправление на IR ковенанта.

        Совпадение проверяется по всей области, включая напечатанное значение: если
        из документа пришло не то, что объявлено опечаткой, подставлять исправленное
        число нельзя — изменился либо документ, либо его разбор.
        """
        erratum = self.for_covenant(ir.borrower_id, ir.covenant_id)
        if erratum is None:
            return ir, None

        documented = ir.comparison.threshold
        if documented != erratum.documented_value:
            raise ErrataError(
                f"{erratum.id}: в {ir.borrower_id}/{ir.covenant_id} ожидался порог "
                f"{erratum.documented_value}, а из документа пришёл {documented}. "
                f"Исправление не применено."
            )

        corrected = ir.model_copy(
            update={
                "comparison": ir.comparison.model_copy(
                    update={"threshold": erratum.corrected_value}
                )
            }
        )
        return corrected, ErratumApplication(
            erratum_id=erratum.id,
            field=erratum.field,
            documented_value=documented,
            applied_value=erratum.corrected_value,
            reason=erratum.reason,
            source=erratum.source,
        )

    def apply_formula(
        self, formula: CovenantFormula
    ) -> tuple[CovenantFormula, ErratumApplication | None]:
        """Наложить то же официальное исправление на исполняемый Formula AST."""
        erratum = self.for_covenant(formula.scenario_id, formula.clause_id)
        if erratum is None:
            return formula, None
        if not isinstance(formula.threshold, Constant):
            raise ErrataError(
                f"{erratum.id}: порог {formula.scenario_id}/{formula.clause_id} "
                "не является константой"
            )
        documented = formula.threshold.value
        if documented != erratum.documented_value:
            raise ErrataError(
                f"{erratum.id}: в {formula.scenario_id}/{formula.clause_id} ожидался порог "
                f"{erratum.documented_value}, а из документа пришёл {documented}. "
                "Исправление не применено."
            )
        corrected = formula.model_copy(
            update={"threshold": Constant(value=erratum.corrected_value)}
        )
        return corrected, ErratumApplication(
            erratum_id=erratum.id,
            field=erratum.field,
            documented_value=documented,
            applied_value=erratum.corrected_value,
            reason=erratum.reason,
            source=erratum.source,
        )
