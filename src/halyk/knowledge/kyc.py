"""Разбор досье KYC: кто считается связанной стороной этого заёмщика.

Порог доли объявлен в тексте самого досье и у каждого заёмщика свой. Брать его
константой нельзя: в открытом наборе встречаются значения от 20 до 40 процентов, и
в каждом досье есть контрагент чуть ниже порога.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

_THRESHOLD = re.compile(
    r"владеет\s+(?P<share>\d+(?:[.,]\d+)?)\s*%\s+и\s+более"
    r"|holds\s+(?P<share_en>\d+(?:[.,]\d+)?)\s*%\s+or\s+more",
    re.IGNORECASE,
)
_HOLDING = re.compile(r"^(?P<name>\S[^\n]*?)\s*\n\s*(?P<share>\d+(?:[.,]\d+)?)\s*%\s*$", re.M)
# Из текстового слоя PDF таблица приходит двумя строками на контрагента, а из OCR —
# строкой Markdown. Это одна и та же таблица, поэтому разбираем оба начертания:
# иначе досье, попавшее под распознавание, осталось бы без порога.
_MARKDOWN_HOLDING = re.compile(
    r"^\|\s*(?P<name>[^|\n]*?)\s*\|\s*(?P<share>\d+(?:[.,]\d+)?)\s*%\s*\|", re.M
)
_TABLE_START = ("Доля голосующих прав", "Share of voting rights", "Voting rights")
_TABLE_END = ("Организации, в которых", "Entities in which", "Organisations in which")

# Второе раскрытие в тех же досье: какая часть активов дочерней организации в залоге.
# Оно устроено так же, но отвечает на другой вопрос, поэтому разбирается отдельно.
_COVERAGE = re.compile(
    r"доля активов в залоге ниже\s+(?P<share>\d+(?:[.,]\d+)?)\s*%"
    r"|share of pledged assets (?:is )?below\s+(?P<share_en>\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)
_COVERAGE_START = ("Доля активов в залоге", "Share of pledged assets", "Pledged assets")
_COVERAGE_END = ("Дочерние организации, у которых", "Subsidiaries whose", "Subsidiaries in which")

# Юридические формы пишутся по-разному в досье и в реестре: «Aktau Holdings LLP»
# против «Aktau Holdings L.L.P.», «Ertis Capital, LLP» с запятой.
_PUNCTUATION = re.compile(r"[.,«»\"'()]")
_SPACES = re.compile(r"\s+")


class KycError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RelatedPartyPolicy:
    """Правило отнесения контрагента к связанным сторонам для одного заёмщика."""

    threshold: Decimal
    holdings: tuple[tuple[str, Decimal], ...]

    @property
    def related_parties(self) -> tuple[str, ...]:
        return tuple(name for name, share in self.holdings if share >= self.threshold)

    @property
    def near_miss(self) -> tuple[tuple[str, Decimal], ...]:
        """Контрагенты под порогом. Держим отдельно — это заготовленные ловушки."""
        return tuple((name, share) for name, share in self.holdings if share < self.threshold)


@dataclass(frozen=True, slots=True)
class CollateralPolicy:
    """Периметр обеспечения: чьи активы в залоге и с какой доли организация в нём."""

    threshold: Decimal
    coverage: tuple[tuple[str, Decimal], ...]

    @property
    def unrestricted(self) -> tuple[tuple[str, Decimal], ...]:
        return tuple((name, share) for name, share in self.coverage if share < self.threshold)


def normalise_counterparty(name: str) -> str:
    """Форма названия для сопоставления досье с реестром.

    NFKC идёт первым: неразрывный пробел и полноширинные знаки приходят из PDF
    неотличимо от обычных, и без приведения `Aktau Holdings` не совпало бы с
    `Aktau Holdings`. Юридическая форма при этом сохраняется — она отбрасывается
    только как пунктуация, поэтому `LLP` и `JSC` продолжают различать компании.
    """
    folded = unicodedata.normalize("NFKC", name)
    return _SPACES.sub(" ", _PUNCTUATION.sub("", folded)).strip().casefold()


def _share(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ".")) / 100


def _declared_share(match: re.Match[str]) -> Decimal:
    """Доля из той ветви шаблона, которая сработала: русской или английской."""
    raw = match.group("share") or match.group("share_en")
    return _share(raw)


def _section(text: str, starts: tuple[str, ...], ends: tuple[str, ...]) -> str | None:
    """Кусок текста между заголовком таблицы и следующим разделом.

    Заголовки перечислены на обоих языках, и берётся первый найденный: документ
    написан на одном языке целиком, поэтому конкурировать им не за что.
    """
    for start in starts:
        if (begin := text.find(start)) < 0:
            continue
        tail = begin + len(start)
        finish = min(
            (found for end in ends if (found := text.find(end, tail)) >= 0), default=len(text)
        )
        return text[tail:finish]
    return None


def _read_holdings(block: str) -> tuple[tuple[str, Decimal], ...]:
    for pattern in (_HOLDING, _MARKDOWN_HOLDING):
        found = tuple(
            (item.group("name").strip(), _share(item.group("share")))
            for item in pattern.finditer(block)
        )
        if found:
            return found
    return ()


def parse_related_party_policy(text: str) -> RelatedPartyPolicy:
    match = _THRESHOLD.search(text)
    if match is None:
        raise KycError("В досье не объявлен порог доли для связанных сторон")

    block = _section(text, _TABLE_START, _TABLE_END)
    if block is None:
        raise KycError("В досье не нашлась таблица долей участия")

    holdings = _read_holdings(block)
    if not holdings:
        raise KycError("Таблица долей участия пуста")

    return RelatedPartyPolicy(threshold=_declared_share(match), holdings=holdings)


def parse_collateral_policy(text: str) -> CollateralPolicy:
    """Разбор таблицы обеспечительного покрытия. Есть не в каждом досье."""
    match = _COVERAGE.search(text)
    if match is None:
        raise KycError("В досье не объявлена граница периметра обеспечения")

    block = _section(text, _COVERAGE_START, _COVERAGE_END)
    if block is None:
        raise KycError("В досье не нашлась таблица обеспечительного покрытия")

    coverage = _read_holdings(block)
    if not coverage:
        raise KycError("Таблица обеспечительного покрытия пуста")
    return CollateralPolicy(threshold=_declared_share(match), coverage=coverage)
