from datetime import date

import pytest

from halyk.ingest.scenarios import ScenarioMap, ScenarioMappingError
from halyk.models.transaction import LedgerRow


def row(txn_id: str, account_id: str) -> LedgerRow:
    return LedgerRow(
        txn_id=txn_id,
        date=date(2025, 1, 1),
        account_id=account_id,
        counterparty="X",
        description="Y",
    )


def test_mapping_is_built_from_transaction_prefixes() -> None:
    rows = [row("TXN-P1-0001", "ACC-7801"), row("TXN-B1-0002", "ACC-7201")]
    mapping = ScenarioMap.build(["P1", "B1"], rows)
    assert mapping.scenario_to_account == {"B1": "ACC-7201", "P1": "ACC-7801"}
    assert mapping.account_to_scenario["ACC-7801"] == "P1"


def test_foreign_accounts_are_ignored() -> None:
    """В реестре сотни посторонних счетов, они не относятся ни к одному сценарию."""
    rows = [row("TXN-P1-0001", "ACC-7801"), row("TXN-9001-0036", "ACC-9001")]
    mapping = ScenarioMap.build(["P1"], rows)
    assert mapping.scenario_to_account == {"P1": "ACC-7801"}


def test_scenario_without_transactions_is_an_error() -> None:
    with pytest.raises(ScenarioMappingError, match="P2"):
        ScenarioMap.build(["P1", "P2"], [row("TXN-P1-0001", "ACC-7801")])


def test_two_accounts_for_one_scenario_is_an_error() -> None:
    rows = [row("TXN-P1-0001", "ACC-7801"), row("TXN-P1-0002", "ACC-7899")]
    with pytest.raises(ScenarioMappingError, match="больше одного счёта"):
        ScenarioMap.build(["P1"], rows)


def test_one_account_for_two_scenarios_is_an_error() -> None:
    rows = [row("TXN-P1-0001", "ACC-7801"), row("TXN-P2-0001", "ACC-7801")]
    with pytest.raises(ScenarioMappingError, match="нескольким сценариям"):
        ScenarioMap.build(["P1", "P2"], rows)
