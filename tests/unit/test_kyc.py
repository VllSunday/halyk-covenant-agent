from decimal import Decimal

import pytest

from halyk.knowledge.kyc import (
    KycError,
    normalise_counterparty,
    parse_collateral_policy,
    parse_related_party_policy,
)

DOSSIER = """Досье «Знай своего клиента» (KYC)
Организация
Доля голосующих прав
Aktau Holdings LLP
34.5%
Kaspi Marine Engineering LLP
18.7%
Ural Crane Works LLP
6.2%
Организации, в которых Группа владеет 20.0% и более голосующих прав, признаются
связанными сторонами для целей Договора.
"""


RECOGNISED_DOSSIER = """## Бенефициарное владение и контроль

| Организация | Доля голосующих прав |
|---|---:|
| "Taraz Holding Group" LLP | 46.8% |
| Taraz Kiln Services LLP | 38.1% |
| Ural Grinding Works LLP | 11.5% |

Организации, в которых Группа владеет 40.0% и более голосующих прав, признаются
связанными сторонами для целей Договора.
"""


def test_threshold_is_read_from_the_dossier() -> None:
    assert parse_related_party_policy(DOSSIER).threshold == Decimal("0.20")


def test_only_holdings_above_the_threshold_count() -> None:
    policy = parse_related_party_policy(DOSSIER)
    assert policy.related_parties == ("Aktau Holdings LLP",)


def test_near_miss_is_kept_separately() -> None:
    # 18.7% при пороге 20% — заготовленная ловушка, её полезно видеть в отчёте.
    policy = parse_related_party_policy(DOSSIER)
    assert policy.near_miss == (
        ("Kaspi Marine Engineering LLP", Decimal("0.187")),
        ("Ural Crane Works LLP", Decimal("0.062")),
    )


def test_threshold_differs_between_borrowers() -> None:
    other = DOSSIER.replace("владеет 20.0%", "владеет 40.0%")
    assert parse_related_party_policy(other).related_parties == ()


def test_recognised_table_is_read_the_same_way() -> None:
    """Досье из OCR приходит таблицей Markdown, а не двумя строками на контрагента.

    Отсканированные досье есть и в открытом наборе, и после распознавания их разбор
    не должен зависеть от того, как таблица легла в текст.
    """
    policy = parse_related_party_policy(RECOGNISED_DOSSIER)
    assert policy.threshold == Decimal("0.40")
    assert policy.related_parties == ('"Taraz Holding Group" LLP',)
    assert policy.near_miss == (
        ("Taraz Kiln Services LLP", Decimal("0.381")),
        ("Ural Grinding Works LLP", Decimal("0.115")),
    )


def test_separator_row_is_not_taken_for_a_holding() -> None:
    assert len(parse_related_party_policy(RECOGNISED_DOSSIER).holdings) == 3


COVERAGE = """**Обеспечительное покрытие дочерних организаций**

| Дочерняя организация | Доля активов в залоге |
|---|---|
| Zhezkazgan Conveyor Assets LLP | 87.6% |
| Zhezkazgan Processing Holdings LLP | 11.4% |

Дочерние организации, у которых доля активов в залоге ниже 50.0%, находятся вне периметра
обеспечения и для целей Договора рассматриваются как неограниченные.
"""


def test_collateral_coverage_is_a_separate_disclosure() -> None:
    """В досье две таблицы долей, и они отвечают на разные вопросы.

    Голосующие права определяют связанную сторону, доля активов в залоге — периметр
    обеспечения. Разбирать их одним правилом нельзя.
    """
    policy = parse_collateral_policy(COVERAGE)
    assert policy.threshold == Decimal("0.50")
    assert policy.coverage == (
        ("Zhezkazgan Conveyor Assets LLP", Decimal("0.876")),
        ("Zhezkazgan Processing Holdings LLP", Decimal("0.114")),
    )
    assert [name for name, _ in policy.unrestricted] == ["Zhezkazgan Processing Holdings LLP"]


def test_dossier_without_coverage_table_is_an_error() -> None:
    with pytest.raises(KycError):
        parse_collateral_policy(DOSSIER)


def test_missing_threshold_is_an_error() -> None:
    with pytest.raises(KycError):
        parse_related_party_policy("Досье без раскрытия долей")


def test_missing_table_is_an_error() -> None:
    with pytest.raises(KycError):
        parse_related_party_policy("Группа владеет 20.0% и более голосующих прав")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Aktau Holdings LLP", "Aktau Holdings L.L.P."),
        ("Ertis Capital, LLP", "Ertis Capital LLP"),
        ('"Turan Capital" LLP', "Turan Capital LLP"),
        ("Ulytau  Capital   LLP.", "ulytau capital llp"),
    ],
)
def test_counterparty_normalisation_bridges_dossier_and_ledger(left: str, right: str) -> None:
    assert normalise_counterparty(left) == normalise_counterparty(right)


def test_normalisation_keeps_different_companies_apart() -> None:
    assert normalise_counterparty("Shymkent Refinery JSC") != normalise_counterparty(
        "Shymkent Refinery Services JSC"
    )


def test_threshold_declared_in_words() -> None:
    """Порог называют и числом, и прописью — это одно и то же правило."""
    text = (
        "Организация\nДоля голосующих прав\n"
        "Syrdarya Capital L.L.P.\n32.1%\nTurkestan Rail Operations LLP\n24.6%\n"
        "Организации, в которых Группе принадлежит не менее одной четверти "
        "голосующих прав, признаются связанными сторонами для целей Договора."
    )
    policy = parse_related_party_policy(text)
    assert policy.threshold == Decimal("0.25")
    assert policy.related_parties == ("Syrdarya Capital L.L.P.",)


def test_dossier_without_table_names_related_party_directly() -> None:
    text = (
        "Запись 1. Контрагент «Altyn Capital L.L.P.» классифицирован как "
        "АФФИЛИРОВАННОЕ ЛИЦО Заёмщика. Платежи данному контрагенту признаются "
        "Ограниченными платежами для целей ковенантов."
    )
    assert parse_related_party_policy(text).related_parties == ("Altyn Capital L.L.P.",)


def test_designated_subsidiary_is_listed_but_not_related() -> None:
    """Дочерняя внутри периметра — структура группы, а не связанная сторона.

    В индекс она всё равно попадает: иначе её название читается как незнакомое и
    похожее написание в реестре поднимет ложную тревогу.
    """
    text = (
        '**Entry 1.** Counterparty "Oskemen Rolling Mill LLP" is a designated '
        "RESTRICTED SUBSIDIARY of the Borrower.\n"
        '**Entry 2.** Counterparty "Altai Ore Processing LLP" is a designated '
        "UNRESTRICTED SUBSIDIARY of the Borrower."
    )
    policy = parse_related_party_policy(text)
    assert policy.related_parties == ()
    assert [name for name, _ in policy.holdings] == [
        "Oskemen Rolling Mill LLP",
        "Altai Ore Processing LLP",
    ]


def test_dossier_may_declare_that_there_are_none() -> None:
    """«Не выявлены» — прочитанный ответ, а не пустой разбор."""
    text = (
        "Проверка связанных сторон · Aktobe Cement JSC\n"
        "Связанные стороны среди контрагентов не выявлены."
    )
    assert parse_related_party_policy(text).related_parties == ()
