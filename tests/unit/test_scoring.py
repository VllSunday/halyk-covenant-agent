import json
from decimal import Decimal
from pathlib import Path

import pytest

from halyk.dev.scoring import (
    ExpectedCell,
    GroundTruthError,
    accuracy,
    load_ground_truth,
    score_submission,
)
from halyk.models.submission import Submission

KEY = {
    ("B1", "6.1"): ExpectedCell(status="BREACH", actual=Decimal("1.68"), evidence_txn_id="TXN-1"),
    ("B1", "6.2"): ExpectedCell(
        status="COMPLIANT", actual=Decimal("1284663.42"), evidence_txn_id=None
    ),
}


def answer(**cells: dict[str, object]) -> Submission:
    return Submission.model_validate(
        {"team": "t", "contact_email": "e", "model": "m", "answers": {"B1": cells}}
    )


PERFECT = {
    "6.1": {"status": "BREACH", "actual": "1.68", "evidence_txn_id": "TXN-1"},
    "6.2": {"status": "COMPLIANT", "actual": "1284663.42", "evidence_txn_id": None},
}


def test_key_against_itself_scores_full_marks() -> None:
    report = score_submission(answer(**PERFECT), KEY)
    assert report.total == Decimal("2.00")
    assert report.exact_cells == 2
    assert report.components == {
        "status": Decimal("1.00"),
        "actual": Decimal("0.60"),
        "evidence": Decimal("0.40"),
    }


def test_wrong_status_zeroes_the_whole_cell() -> None:
    spoiled = dict(PERFECT) | {"6.1": PERFECT["6.1"] | {"status": "COMPLIANT"}}
    cell = score_submission(answer(**spoiled), KEY).cells[0]
    assert cell.total == Decimal(0)
    assert "вся ячейка обнулена" in cell.note


def test_missing_cell_scores_zero() -> None:
    report = score_submission(answer(**{"6.1": PERFECT["6.1"]}), KEY)
    missing = next(c for c in report.cells if c.covenant == "6.2")
    assert missing.total == Decimal(0)
    assert missing.note == "ячейка пропущена"


def test_wrong_evidence_costs_only_its_own_weight() -> None:
    spoiled = dict(PERFECT) | {"6.1": PERFECT["6.1"] | {"evidence_txn_id": "TXN-9"}}
    cell = score_submission(answer(**spoiled), KEY).cells[0]
    assert cell.total == Decimal("0.80")
    assert cell.evidence_points == Decimal(0)


def test_empty_key_evidence_decays_with_the_number() -> None:
    # 2.5% ошибки — половина и от 0.30, и от 0.20: на таких ячейках число весит 0.50.
    off = Decimal("1284663.42") * Decimal("1.025")
    spoiled = dict(PERFECT) | {"6.2": PERFECT["6.2"] | {"actual": str(off)}}
    cell = next(c for c in score_submission(answer(**spoiled), KEY).cells if c.covenant == "6.2")
    assert cell.status_points == Decimal("0.50")
    assert cell.actual_points == pytest.approx(Decimal("0.15"), abs=Decimal("0.001"))
    assert cell.evidence_points == pytest.approx(Decimal("0.10"), abs=Decimal("0.001"))


@pytest.mark.parametrize(
    ("value", "expected", "share"),
    [
        ("100", "100", "1"),
        ("102.5", "100", "0.5"),
        ("105", "100", "0"),
        ("200", "100", "0"),
        (None, "100", "0"),
    ],
)
def test_accuracy_scale(value: str | None, expected: str, share: str) -> None:
    measured = Decimal(value) if value is not None else None
    assert accuracy(measured, Decimal(expected)) == Decimal(share)


def test_rounding_to_two_decimals_is_scored_not_the_raw_value() -> None:
    # Скорим ровно то, что уйдёт в файл, вместе с округлением сериализации.
    nearly = dict(PERFECT) | {"6.1": PERFECT["6.1"] | {"actual": "1.6849"}}
    assert score_submission(answer(**nearly), KEY).cells[0].total == Decimal("1.00")


def test_extra_cell_makes_the_report_incomparable() -> None:
    """Лишняя ячейка не должна проходить: итерация идёт по ключу и её не заметит.

    Ложное «полный балл» на скорере, по которому мы принимаем решения о подаче, —
    худшая из возможных ошибок здесь.
    """
    inflated = Submission.model_validate(
        {
            "team": "t",
            "contact_email": "e",
            "model": "m",
            "answers": {
                "B1": PERFECT,
                "EXTRA": {"9.9": {"status": "BREACH", "actual": "1.0", "evidence_txn_id": None}},
            },
        }
    )
    report = score_submission(inflated, KEY)
    assert report.total == Decimal("2.00")
    assert not report.is_comparable
    assert report.unexpected == (("EXTRA", "9.9"),)
    assert report.missing == ()


def test_missing_cell_is_listed_as_well() -> None:
    report = score_submission(answer(**{"6.1": PERFECT["6.1"]}), KEY)
    assert report.missing == (("B1", "6.2"),)
    assert not report.is_comparable


def test_matching_keyset_is_comparable() -> None:
    assert score_submission(answer(**PERFECT), KEY).is_comparable


def test_load_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": {
                    "P1": {
                        "covenants": {
                            "6.1": {"status": "BREACH", "actual": 0.46, "evidence_txn_id": None}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_ground_truth(path)
    assert loaded[("P1", "6.1")] == ExpectedCell(
        status="BREACH", actual=Decimal("0.46"), evidence_txn_id=None
    )


def test_broken_ground_truth_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({"scenarios": {"P1": {}}}), encoding="utf-8")
    with pytest.raises(GroundTruthError):
        load_ground_truth(path)
