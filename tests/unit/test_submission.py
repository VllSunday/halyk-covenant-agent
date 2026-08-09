import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from halyk.models.result import Verdict
from halyk.models.submission import CovenantCell, Submission
from halyk.output.validator import validate_document


def cell(**kwargs: object) -> CovenantCell:
    payload = {"status": "BREACH", "actual": "1.68", "evidence_txn_id": None, **kwargs}
    return CovenantCell.model_validate(payload)


def test_verdict_values_match_the_key() -> None:
    assert Verdict.COMPLIANT == "COMPLIANT"
    assert Verdict.BREACH == "BREACH"


@pytest.mark.parametrize("status", ["compliant", "violated", "Compliant", "BREACHED"])
def test_wrong_status_spelling_is_rejected(status: str) -> None:
    with pytest.raises(ValidationError):
        CovenantCell.model_validate({"status": status, "actual": "1.0", "evidence_txn_id": None})


def test_evidence_key_is_required_even_when_empty() -> None:
    # null — это значение третьего компонента, а не основание выкинуть ключ.
    with pytest.raises(ValidationError):
        CovenantCell.model_validate({"status": "BREACH", "actual": "1.0"})


def test_unknown_field_in_cell_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CovenantCell.model_validate(
            {"status": "BREACH", "actual": "1.0", "evidence_txn_id": None, "junk": 7}
        )


def test_unknown_field_in_document_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Submission.model_validate(
            {
                "team": "t",
                "contact_email": "e",
                "model": "m",
                "answers": {},
                "notes": "лишнее",
            }
        )


def test_actual_is_serialised_as_a_number_with_two_decimals() -> None:
    payload = cell(actual="1284663.4231").model_dump(mode="json")
    assert payload["actual"] == 1284663.42
    assert isinstance(payload["actual"], float)


def test_actual_keeps_full_precision_until_serialisation() -> None:
    # Вердикт считается по сырому значению: округлив раньше, можно перевернуть
    # сравнение с порогом на границе.
    assert cell(actual="0.4249").actual == Decimal("0.4249")


def test_negative_actual_is_rejected() -> None:
    # В реестре расходы записаны со знаком минус, потерянный abs уехал бы в ответ.
    with pytest.raises(ValidationError):
        cell(actual="-283664.18")


def test_document_shape_matches_the_template() -> None:
    submission = Submission(
        team="halyk",
        contact_email="team@example.com",
        model="claude-opus-5",
        answers={"P1": {"6.1": cell(actual="0.46", evidence_txn_id=None)}},
    )
    document = json.loads(submission.model_dump_json())
    assert document == {
        "team": "halyk",
        "contact_email": "team@example.com",
        "model": "claude-opus-5",
        "answers": {"P1": {"6.1": {"status": "BREACH", "actual": 0.46, "evidence_txn_id": None}}},
    }


def test_serialisation_schema_keeps_the_lower_bound() -> None:
    """Схема обязана ловить то же, что и модель.

    `halyk validate` без шаблона сверяет файл только со схемой, поэтому граница из
    Field(ge=0) должна доезжать до неё, а не теряться в своём сериализаторе.
    """
    schema = Submission.model_json_schema(mode="serialization")
    document = {
        "team": "t",
        "contact_email": "e",
        "model": "m",
        "answers": {"P1": {"6.1": {"status": "BREACH", "actual": -1.0, "evidence_txn_id": None}}},
    }
    # Ячейка описана объединением «посчитанная или пустая», поэтому адрес нарушения
    # указывает на неё целиком: отрицательное число не подходит ни под один вариант.
    assert [issue.location for issue in validate_document(document, schema)] == ["answers/P1/6.1"]


def test_serialisation_schema_requires_all_three_components() -> None:
    schema = Submission.model_json_schema(mode="serialization")
    document = {
        "team": "t",
        "contact_email": "e",
        "model": "m",
        "answers": {"P1": {"6.1": {"status": "BREACH", "actual": 1.0}}},
    }
    assert validate_document(document, schema)


def test_cell_keys_are_pairs() -> None:
    submission = Submission(
        team="t",
        contact_email="e",
        model="m",
        answers={"P1": {"6.1": cell(), "6.2": cell()}, "B1": {"6.1": cell()}},
    )
    assert submission.cell_keys() == {("P1", "6.1"), ("P1", "6.2"), ("B1", "6.1")}
