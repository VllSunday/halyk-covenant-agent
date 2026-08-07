from datetime import date
from pathlib import Path

import pytest

from halyk.ingest.ledger import LedgerError, read_ledger
from halyk.money import Currency

HEADER = "txn_id,date,account_id,counterparty,description,amount,currency\n"
ROWS = (
    "TXN-P1-0031,2025-07-09,ACC-7801,Aktau Holdings L.L.P.,Management advisory,-283664.18,USD\n"
    "TXN-P3-0024,2025-05-02,ACC-7803,Rheinland GmbH,Catalyst servicing,-612884.25,EUR\n"
    "TXN-P7-0033,2025-04-21,ACC-7807,State Revenue Committee,Mineral tax,,USD\n"
)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ledger.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_rows_are_parsed(tmp_path: Path) -> None:
    rows = read_ledger(write(tmp_path, ROWS))
    assert len(rows) == 3
    assert rows[0].date == date(2025, 7, 9)
    assert rows[0].amount is not None
    assert rows[0].amount.minor == -28366418


def test_currency_travels_with_the_amount(tmp_path: Path) -> None:
    rows = read_ledger(write(tmp_path, ROWS))
    assert rows[1].amount is not None
    assert rows[1].amount.currency is Currency.EUR


def test_empty_amount_stays_empty(tmp_path: Path) -> None:
    """Пустая сумма — часть задачи, её значение раскрыто в другом документе.

    Ноль здесь стёр бы разницу между «нет данных» и «нулевой оборот».
    """
    rows = read_ledger(write(tmp_path, ROWS))
    assert rows[2].amount is None


def test_scenario_id_comes_from_the_transaction_id(tmp_path: Path) -> None:
    rows = read_ledger(write(tmp_path, ROWS))
    assert [row.scenario_id for row in rows] == ["P1", "P3", "P7"]


def test_missing_column_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("txn_id,date,account_id\nTXN-P1-0001,2025-01-01,ACC-7801\n", encoding="utf-8")
    with pytest.raises(LedgerError):
        read_ledger(path)


def test_bad_date_is_reported(tmp_path: Path) -> None:
    body = "TXN-P1-0001,09.07.2025,ACC-7801,X,Y,-1.00,USD\n"
    with pytest.raises(LedgerError):
        read_ledger(write(tmp_path, body))


def test_unknown_currency_is_reported(tmp_path: Path) -> None:
    body = "TXN-P1-0001,2025-07-09,ACC-7801,X,Y,-1.00,GBP\n"
    with pytest.raises(LedgerError):
        read_ledger(write(tmp_path, body))
