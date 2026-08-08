"""Факты, вычитанные из документов, вместе с адресом источника.

Слой отделён от корректировок реестра намеренно. Часть фактов ничего не меняет в
операциях, а входит в формулу напрямую: порог разовых статей, покрытие дочерних
организаций, обязательство, которого в бухгалтерской книге вовсе нет. Превращать
такое в синтетическую проводку значило бы получить сумму, которой не существует ни
в одном документе, и потерять объяснимость ответа.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from halyk.knowledge.kyc import RelatedPartyPolicy
from halyk.models.source import SourceRef
from halyk.money import Money


class FactKind(StrEnum):
    RELATED_PARTY_POLICY = "related_party_policy"
    COLLATERAL_COVERAGE = "collateral_coverage"
    ONE_OFF_ITEM = "one_off_item"
    ONE_OFF_POLICY = "one_off_policy"
    AGGREGATE_OBLIGATION = "aggregate_obligation"
    FX_SETTLEMENT = "fx_settlement"
    PPE_ROLL_FORWARD = "ppe_roll_forward"


class Share(BaseModel):
    """Доля контрагента: голосующие права или активы в залоге."""

    model_config = ConfigDict(frozen=True)

    counterparty: str
    share: Decimal = Field(ge=0, le=1)


class FactBase(BaseModel):
    """Общая часть: чей это факт и откуда взят.

    Тип объявляется в каждой модели отдельным `Literal` — он не просто подпись, а
    признак, по которому факт разбирается обратно из артефакта. Общее поле
    позволяло бы собрать разовую статью с чужим типом, и союз перестал бы что-либо
    гарантировать потребителю.
    """

    model_config = ConfigDict(frozen=True)

    account_id: str
    source: SourceRef

    def record(self) -> dict[str, Any]:
        """Строка для JSONL-артефакта."""
        return self.model_dump(mode="json")


class RelatedPartyPolicyFact(FactBase):
    """Порог отнесения к связанным сторонам и доли из досье KYC."""

    kind: Literal[FactKind.RELATED_PARTY_POLICY] = FactKind.RELATED_PARTY_POLICY
    threshold: Decimal = Field(ge=0, le=1)
    holdings: tuple[Share, ...]

    @property
    def related_parties(self) -> tuple[str, ...]:
        return tuple(h.counterparty for h in self.holdings if h.share >= self.threshold)

    def as_policy(self) -> RelatedPartyPolicy:
        """Обратный переход к разобранной политике для построения индекса.

        Факт хранится в артефакте прогона, а индекс строится из политики: без этого
        моста пришлось бы либо разбирать досье второй раз, либо тащить политику мимо
        артефакта, и они разошлись бы при первой же правке разбора.
        """
        return RelatedPartyPolicy(
            threshold=self.threshold,
            holdings=tuple((h.counterparty, h.share) for h in self.holdings),
        )


class CollateralCoverageFact(FactBase):
    """Доля активов дочерних организаций в залоге и граница периметра обеспечения."""

    kind: Literal[FactKind.COLLATERAL_COVERAGE] = FactKind.COLLATERAL_COVERAGE
    threshold: Decimal = Field(ge=0, le=1)
    subsidiaries: tuple[Share, ...]

    @property
    def unrestricted(self) -> tuple[str, ...]:
        return tuple(s.counterparty for s in self.subsidiaries if s.share < self.threshold)


class OneOffItemFact(FactBase):
    """Разовая статья из примечаний. К EBITDA прибавляется не всякая — см. порог."""

    kind: Literal[FactKind.ONE_OFF_ITEM] = FactKind.ONE_OFF_ITEM
    description: str
    counterparty: str
    amount: Money


class OneOffPolicyFact(FactBase):
    """Минимальная сумма, с которой статья считается разовой."""

    kind: Literal[FactKind.ONE_OFF_POLICY] = FactKind.ONE_OFF_POLICY
    minimum: Money


class AggregateObligationFact(FactBase):
    """Обязательство, раскрытое в примечаниях и не отражённое отдельной операцией."""

    kind: Literal[FactKind.AGGREGATE_OBLIGATION] = FactKind.AGGREGATE_OBLIGATION
    description: str
    amount: Money


class FxSettlementFact(FactBase):
    """Пара «счёт — платёж», по которой восстанавливается курс пересчёта.

    Отдельной таблицы курсов в датасете нет: примечания раскрывают конкретный
    расчёт, и курс выводится из него. Хранить нужно обе суммы, а не только
    отношение, иначе в объяснении ответа неоткуда взять исходные числа.
    """

    kind: Literal[FactKind.FX_SETTLEMENT] = FactKind.FX_SETTLEMENT
    counterparty: str
    invoiced: Money
    settled: Money

    @property
    def rate(self) -> Decimal:
        return self.settled.to_decimal() / self.invoiced.to_decimal()


class PpeRollForwardFact(FactBase):
    """Движение основных средств, из которого выводятся капитальные затраты.

    Отдельной строки «capital expenditure» в отчётности нет, поэтому хранятся все
    раскрытые движения, а не одно выведенное число: по ним видно, из чего получились
    капзатраты, и на них же проверяется, что тождество замкнулось.
    """

    kind: Literal[FactKind.PPE_ROLL_FORWARD] = FactKind.PPE_ROLL_FORWARD
    opening: Money
    closing: Money
    depreciation: Money
    disposals: Money
    # Страница со связью документа с заёмщиком отличается от страницы с числами, а
    # объяснение ответа должно показывать обе.
    supporting: tuple[SourceRef, ...] = ()

    @property
    def additions(self) -> Decimal:
        """`closing = opening + additions − depreciation − disposals`."""
        return (
            self.closing.to_decimal()
            - self.opening.to_decimal()
            + self.depreciation.to_decimal()
            + self.disposals.to_decimal()
        )


Fact = Annotated[
    RelatedPartyPolicyFact
    | CollateralCoverageFact
    | OneOffItemFact
    | OneOffPolicyFact
    | AggregateObligationFact
    | FxSettlementFact
    | PpeRollForwardFact,
    Field(discriminator="kind"),
]

FactAdapter: TypeAdapter[Fact] = TypeAdapter(Fact)
