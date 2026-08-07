import json
from pathlib import Path

import pytest

from halyk.models.submission import CovenantCell, Submission
from halyk.output.template import SubmissionTemplate, TemplateError

BLANK = {"status": None, "actual": None, "evidence_txn_id": None}


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    path = tmp_path / "submission_template.json"
    path.write_text(
        json.dumps(
            {
                "team": "",
                "contact_email": "",
                "model": "",
                "answers": {
                    "P1": {"6.1": BLANK, "6.2": BLANK},
                    "B1": {"6.1": BLANK},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def answer() -> CovenantCell:
    return CovenantCell.model_validate(
        {"status": "COMPLIANT", "actual": "1.0", "evidence_txn_id": None}
    )


def submission(answers: dict[str, dict[str, CovenantCell]], **fields: str) -> Submission:
    return Submission.model_validate(
        {
            "team": fields.get("team", "halyk"),
            "contact_email": fields.get("contact_email", "team@example.com"),
            "model": fields.get("model", "claude-opus-5"),
            "answers": answers,
        }
    )


def test_template_keeps_order_and_scenarios(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    assert template.cells == (("P1", "6.1"), ("P1", "6.2"), ("B1", "6.1"))
    assert template.scenarios == ("P1", "B1")


def test_complete_answer_passes(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    filled = submission({"P1": {"6.1": answer(), "6.2": answer()}, "B1": {"6.1": answer()}})
    assert template.check(filled) == []


def test_missing_cell_is_reported(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    issues = template.check(submission({"P1": {"6.1": answer()}, "B1": {"6.1": answer()}}))
    assert [(i.location, i.message) for i in issues] == [("answers/P1/6.2", "ячейка пропущена")]


def test_unexpected_cell_is_reported(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    filled = submission(
        {
            "P1": {"6.1": answer(), "6.2": answer(), "6.3": answer()},
            "B1": {"6.1": answer()},
        }
    )
    issues = template.check(filled)
    assert [(i.location, i.message) for i in issues] == [("answers/P1/6.3", "ячейки нет в шаблоне")]


def test_unknown_scenario_is_reported(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    filled = submission(
        {"P1": {"6.1": answer(), "6.2": answer()}, "B1": {"6.1": answer()}, "T9": {"6.1": answer()}}
    )
    assert [i.location for i in template.check(filled)] == ["answers/T9/6.1"]


def test_empty_header_fields_are_reported(template_path: Path) -> None:
    template = SubmissionTemplate.load(template_path)
    filled = submission(
        {"P1": {"6.1": answer(), "6.2": answer()}, "B1": {"6.1": answer()}},
        team="   ",
        model="",
    )
    assert [i.location for i in template.check(filled)] == ["team", "model"]


def test_broken_template_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"answers": {}}), encoding="utf-8")
    with pytest.raises(TemplateError):
        SubmissionTemplate.load(path)
