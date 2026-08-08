"""Разрешение сущностей: связан ли контрагент операции с заёмщиком.

Этап отделён от расчёта намеренно. Сопоставление названий — это интерпретация
данных, а не арифметика: смешав их, исполнитель начал бы решать, кто такая
`Aktau Holdings L.L.P.`, посреди вычисления отношения. Здесь строится индекс, там
только сверяются ключи.

Отсутствие операций у известной связанной стороны ошибкой не является: ноль
платежей — обычный исход года. Ошибка — невозможность решить, связана ли конкретная
подходящая операция.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from enum import StrEnum

from halyk.knowledge.kyc import RelatedPartyPolicy, normalise_counterparty

# Порог подозрения на опечатку. Автоматически по нему ничего не связывается: он лишь
# отделяет «в досье такой компании нет» от «есть почти такая, и надо смотреть
# глазами».
#
# Значение выбрано по замеру на двух реальных случаях набора. Опечатка в одну букву
# («Holdinqs» вместо «Holdings») даёт 0.94, а компании-двойники, различающиеся целым
# словом («Shymkent Refinery JSC» и «Shymkent Refinery Services JSC»), — 0.82.
# Граница между ними ловит первое и не трогает второе.
_SUSPICION = 0.90


class RelatedPartyError(ValueError):
    """Связанность операции определить нельзя. Догадка здесь дороже остановки."""


class Relation(StrEnum):
    RELATED = "related"
    UNRELATED = "unrelated"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Match:
    """Решение по одному контрагенту вместе с основанием — уходит в lineage."""

    counterparty: str
    relation: Relation
    rule: str
    canonical: str | None = None
    candidate: str | None = None

    @property
    def is_related(self) -> bool:
        return self.relation is Relation.RELATED


@dataclass(frozen=True, slots=True)
class RelatedPartyIndex:
    """Связанные стороны одного заёмщика в форме, пригодной для точного сравнения."""

    account_id: str
    threshold: Decimal
    canonical: dict[str, str]
    related_keys: frozenset[str]

    @property
    def related_parties(self) -> tuple[str, ...]:
        return tuple(sorted(self.canonical[key] for key in self.related_keys))

    def classify(self, counterparty: str) -> Match:
        """Решение по названию из реестра.

        Незнакомое название — это `UNRELATED`, а не ошибка: в реестре полно
        контрагентов, которых досье и не должно упоминать. Ошибкой становится только
        похожее написание, потому что выбор между «другая компания» и «опечатка в той
        же» здесь сделать нечем.
        """
        if not counterparty or not counterparty.strip():
            return Match(counterparty=counterparty, relation=Relation.UNRESOLVED, rule="blank")

        key = normalise_counterparty(counterparty)
        if key in self.canonical:
            return Match(
                counterparty=counterparty,
                relation=Relation.RELATED if key in self.related_keys else Relation.UNRELATED,
                rule="normalised",
                canonical=self.canonical[key],
            )

        if (near := self._nearest(key)) is not None:
            return Match(
                counterparty=counterparty,
                relation=Relation.UNRESOLVED,
                rule="near",
                candidate=near,
            )
        return Match(counterparty=counterparty, relation=Relation.UNRELATED, rule="unknown")

    def _nearest(self, key: str) -> str | None:
        """Ближайшее название из досье, если оно подозрительно близко.

        Сравнение идёт по всем ключам в отсортированном порядке: при равном сходстве
        ответ не должен зависеть от того, в каком порядке досье перечислило компании.
        """
        best: tuple[float, str] | None = None
        for candidate in sorted(self.canonical):
            ratio = SequenceMatcher(None, key, candidate).ratio()
            if ratio >= _SUSPICION and (best is None or ratio > best[0]):
                best = (ratio, self.canonical[candidate])
        return best[1] if best else None

    def require(self, match: Match) -> Match:
        """Проверка в точке применения фильтра.

        Вызывается исполнителем только для операций, прошедших остальные условия
        селектора: неразрешённый контрагент у строки, которая в расчёт всё равно не
        входит, останавливать прогон не должен.
        """
        if match.relation is not Relation.UNRESOLVED:
            return match
        if match.rule == "blank":
            raise RelatedPartyError(
                f"{self.account_id}: у операции пустой контрагент, а фильтр требует "
                f"решения о связанности"
            )
        raise RelatedPartyError(
            f"{self.account_id}: контрагент {match.counterparty!r} не найден в досье, "
            f"но похож на {match.candidate!r}. Связывать по сходству нельзя — "
            f"сверьте написание."
        )


def build_index(account_id: str, policy: RelatedPartyPolicy | None) -> RelatedPartyIndex:
    """Индекс по досье заёмщика.

    Отсутствие досье — остановка: без порога и списка долей любая операция получила
    бы `UNRELATED`, и ковенант по связанным сторонам молча посчитался бы по всему
    реестру.
    """
    if policy is None:
        raise RelatedPartyError(
            f"{account_id}: досье KYC не разобрано, связанность определять нечем"
        )

    canonical: dict[str, str] = {}
    for name, _ in policy.holdings:
        key = normalise_counterparty(name)
        if (seen := canonical.get(key)) is not None and seen != name:
            raise RelatedPartyError(
                f"{account_id}: {name!r} и {seen!r} после нормализации дали один ключ "
                f"{key!r}. Это разные компании либо ошибка разбора досье."
            )
        canonical[key] = name

    # Порог сравнивается тем же оператором, что объявлен в досье: «владеет N% и
    # более» — это `>=`, и доля ровно на границе связывает контрагента.
    related = frozenset(
        normalise_counterparty(name) for name, share in policy.holdings if share >= policy.threshold
    )
    return RelatedPartyIndex(
        account_id=account_id,
        threshold=policy.threshold,
        canonical=canonical,
        related_keys=related,
    )
