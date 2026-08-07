"""Официальные исправления к текстам договоров."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from halyk.knowledge.errata import ErrataError, ErrataRegistry
from halyk.models.covenant import (
    Aggregation,
    Comparison,
    CovenantIR,
    EvidenceRule,
    Operator,
    Period,
    Unit,
)
from halyk.models.source import SourceRef

SOURCE = SourceRef(file_hash="a" * 64, file_name="agreement.pdf", page=5)


def covenant(borrower_id: str, covenant_id: str, threshold: str) -> CovenantIR:
    return CovenantIR(
        borrower_id=borrower_id,
        covenant_id=covenant_id,
        clause_id="6.3",
        metric="related_party_payments_to_revenue",
        measurement_period=Period(start=date(2025, 1, 1), end=date(2025, 12, 31)),
        aggregation=Aggregation.RATIO,
        comparison=Comparison(operator=Operator.LE, threshold=Decimal(threshold), unit=Unit.RATIO),
        evidence_rule=EvidenceRule.NOT_APPLICABLE,
        source_refs=(SOURCE,),
        confidence=0.9,
    )


@pytest.fixture(scope="module")
def registry() -> ErrataRegistry:
    return ErrataRegistry.load()


def test_registry_is_loaded_with_its_digest(registry: ErrataRegistry) -> None:
    assert registry.digest is not None
    assert len(registry.digest) == 64
    assert [e.id for e in registry.entries] == ["E-001"]


def test_registry_is_found_from_any_working_directory(
    registry: ErrataRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Команда, запущенная не из корня репозитория, обязана видеть тот же реестр.

    Иначе официальное исправление молча исчезает из прогона, и P4/6.3 считается
    по напечатанному порогу — с виду успешно.
    """
    monkeypatch.chdir(tmp_path)
    elsewhere = ErrataRegistry.load()
    assert [e.id for e in elsewhere.entries] == ["E-001"]
    assert elsewhere.digest == registry.digest


def test_missing_registry_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ErrataError, match="не найден"):
        ErrataRegistry.load(tmp_path / "нет.toml")


def test_empty_registry_requires_an_explicit_choice() -> None:
    assert ErrataRegistry().entries == ()
    assert ErrataRegistry().digest is None


def test_threshold_of_p4_is_corrected(registry: ErrataRegistry) -> None:
    """В договоре напечатано 0.04, организаторы объявили это опечаткой."""
    corrected, applied = registry.apply(covenant("P4", "6.3", "0.04"))
    assert corrected.comparison.threshold == Decimal("0.045")
    assert applied is not None
    assert applied.documented_value == Decimal("0.04")
    assert applied.erratum_id == "E-001"


def test_application_keeps_both_values_for_the_lineage(registry: ErrataRegistry) -> None:
    _, applied = registry.apply(covenant("P4", "6.3", "0.04"))
    assert applied is not None
    record = applied.record()
    assert record["documented_value"] == "0.04"
    assert record["applied_value"] == "0.045"
    assert record["source"].startswith("Официальное сообщение")


def test_other_borrowers_with_the_same_threshold_are_untouched(
    registry: ErrataRegistry,
) -> None:
    """У P8 в 6.3 тот же порог 0.04, и он настоящий."""
    same, applied = registry.apply(covenant("P8", "6.3", "0.04"))
    assert same.comparison.threshold == Decimal("0.04")
    assert applied is None


def test_other_covenants_of_the_same_borrower_are_untouched(registry: ErrataRegistry) -> None:
    same, applied = registry.apply(covenant("P4", "6.2", "0.04"))
    assert same.comparison.threshold == Decimal("0.04")
    assert applied is None


def test_unexpected_documented_value_stops_the_run(registry: ErrataRegistry) -> None:
    """Если из PDF пришло не то, что объявлено опечаткой, подставлять нельзя.

    Значит, изменился документ или его разбор, и молча получить нужное число —
    ровно тот случай, ради которого исправления и вынесены из кода.
    """
    with pytest.raises(ErrataError, match="ожидался порог"):
        registry.apply(covenant("P4", "6.3", "0.07"))


def test_registry_rejects_two_corrections_of_one_field(tmp_path: Path) -> None:
    path = tmp_path / "errata.toml"
    entry = """
[[erratum]]
id = "{id}"
scenario = "P4"
covenant = "6.3"
field = "threshold"
documented_value = "0.04"
corrected_value = "{value}"
announced_on = 2026-08-07
reason = "проверка"
source = "проверка"
"""
    path.write_text(
        entry.format(id="E-001", value="0.045") + entry.format(id="E-002", value="0.05"),
        encoding="utf-8",
    )
    with pytest.raises(ErrataError):
        ErrataRegistry.load(path)
