"""Сопоставление контрагентов досье с реестром."""

from __future__ import annotations

from decimal import Decimal

import pytest

from halyk.knowledge.kyc import RelatedPartyPolicy
from halyk.knowledge.related_party import (
    RelatedPartyError,
    Relation,
    build_index,
)


def policy(*holdings: tuple[str, str], threshold: str = "0.20") -> RelatedPartyPolicy:
    return RelatedPartyPolicy(
        threshold=Decimal(threshold),
        holdings=tuple((name, Decimal(share)) for name, share in holdings),
    )


def index(*holdings: tuple[str, str], threshold: str = "0.20"):  # type: ignore[no-untyped-def]
    return build_index("ACC-7801", policy(*holdings, threshold=threshold))


def test_legal_form_spelling_matches() -> None:
    """`Aktau Holdings LLP` в досье и `Aktau Holdings L.L.P.` в реестре — одна компания."""
    match = index(("Aktau Holdings LLP", "0.35")).classify("Aktau Holdings L.L.P.")
    assert match.relation is Relation.RELATED
    assert match.canonical == "Aktau Holdings LLP"
    assert match.rule == "normalised"


def test_comma_and_spacing_do_not_matter() -> None:
    match = index(("Ertis Capital, LLP", "0.51")).classify("Ertis  Capital LLP")
    assert match.relation is Relation.RELATED


def test_non_breaking_space_matches() -> None:
    """Из PDF приходит неразрывный пробел, и глазами он неотличим от обычного."""
    match = index(("Aktau Holdings LLP", "0.35")).classify("Aktau Holdings LLP")
    assert match.relation is Relation.RELATED


def test_related_party_without_transactions_is_not_an_error() -> None:
    """Ноль платежей связанной стороне — обычный исход года, а не сбой разбора."""
    built = index(("Aktau Holdings LLP", "0.35"))
    assert built.related_parties == ("Aktau Holdings LLP",)
    # Индекс построен, ни одной операции не сопоставлено — и это молча допустимо.
    assert built.classify("Northwind Catering").relation is Relation.UNRELATED


def test_counterparty_absent_from_the_dossier_is_unrelated() -> None:
    match = index(("Aktau Holdings LLP", "0.35")).classify("Northwind Catering")
    assert match.relation is Relation.UNRELATED
    assert match.rule == "unknown"


def test_holding_below_the_threshold_is_unrelated() -> None:
    """Доля 18.7 при пороге 20 — заготовленная ловушка набора."""
    match = index(("Kaspi Mining LLP", "0.187")).classify("Kaspi Mining LLP")
    assert match.relation is Relation.UNRELATED


def test_share_exactly_at_the_threshold_is_related() -> None:
    """Досье говорит «владеет N% и более»: граница включительная."""
    match = index(("Kaspi Mining LLP", "0.20")).classify("Kaspi Mining LLP")
    assert match.relation is Relation.RELATED


def test_only_and_exclude_split_the_same_rows() -> None:
    """ONLY и EXCLUDE обязаны быть дополнением друг друга на решённых контрагентах."""
    built = index(("Aktau Holdings LLP", "0.35"), ("Kaspi Mining LLP", "0.187"))
    names = ["Aktau Holdings L.L.P.", "Kaspi Mining LLP", "Northwind Catering"]
    matches = [built.classify(name) for name in names]

    related = {m.counterparty for m in matches if m.is_related}
    unrelated = {m.counterparty for m in matches if m.relation is Relation.UNRELATED}
    assert related == {"Aktau Holdings L.L.P."}
    assert related | unrelated == set(names)
    assert not related & unrelated


def test_blank_counterparty_is_unresolved() -> None:
    match = index(("Aktau Holdings LLP", "0.35")).classify("   ")
    assert match.relation is Relation.UNRESOLVED
    assert match.rule == "blank"


def test_blank_counterparty_stops_the_run_when_the_filter_needs_it() -> None:
    built = index(("Aktau Holdings LLP", "0.35"))
    with pytest.raises(RelatedPartyError, match="пустой контрагент"):
        built.require(built.classify(""))


def test_similar_spelling_is_never_linked_automatically() -> None:
    """Похожее написание — повод остановиться, а не связать."""
    built = index(("Aktau Holdings LLP", "0.35"))
    match = built.classify("Aktau Holdinqs LLP")
    assert match.relation is Relation.UNRESOLVED
    assert match.rule == "near"
    assert match.candidate == "Aktau Holdings LLP"
    with pytest.raises(RelatedPartyError, match="похож"):
        built.require(match)


def test_twin_companies_stay_apart() -> None:
    """`Shymkent Refinery JSC` и `Shymkent Refinery Services JSC` — разные заёмщики."""
    match = index(("Shymkent Refinery JSC", "0.51")).classify("Shymkent Refinery Services JSC")
    assert match.relation is not Relation.RELATED


def test_collision_after_normalisation_is_an_error() -> None:
    """Два разных названия с одним ключом означают ошибку разбора досье."""
    with pytest.raises(RelatedPartyError, match="один ключ"):
        index(("Ertis Capital LLP", "0.51"), ("Ertis Capital, L.L.P.", "0.30"))


def test_missing_dossier_stops_the_run() -> None:
    with pytest.raises(RelatedPartyError, match="досье KYC не разобрано"):
        build_index("ACC-7801", None)


def test_resolved_matches_pass_through_require() -> None:
    built = index(("Aktau Holdings LLP", "0.35"))
    for name in ("Aktau Holdings LLP", "Northwind Catering"):
        match = built.classify(name)
        assert built.require(match) is match


def test_order_of_holdings_does_not_change_anything() -> None:
    forward = index(("Aktau Holdings LLP", "0.35"), ("Kaspi Mining LLP", "0.187"))
    backward = index(("Kaspi Mining LLP", "0.187"), ("Aktau Holdings LLP", "0.35"))
    assert forward.related_parties == backward.related_parties
    for name in ("Aktau Holdings L.L.P.", "Kaspi Mining LLP", "Aktau Holdinqs LLP"):
        assert forward.classify(name).relation is backward.classify(name).relation


def test_lineage_keeps_the_original_spelling() -> None:
    """В объяснении должно быть видно, что пришло из реестра, а что из досье."""
    match = index(("Aktau Holdings LLP", "0.35")).classify("Aktau Holdings L.L.P.")
    assert match.counterparty == "Aktau Holdings L.L.P."
    assert match.canonical == "Aktau Holdings LLP"
    assert match.rule == "normalised"
