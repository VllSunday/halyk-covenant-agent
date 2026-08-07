"""Связь между сценарием из шаблона и счётом заёмщика.

Единственный мост между шаблоном ответа и документами — реестр: идентификатор
операции начинается с `scenario_id`, а сама операция лежит на счёте заёмщика. Это
правило записано в условии задачи, поэтому на него опираемся напрямую.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from halyk.models.transaction import LedgerRow


class ScenarioMappingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScenarioMap:
    scenario_to_account: dict[str, str]
    account_to_scenario: dict[str, str]
    ambiguous: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(self.scenario_to_account)

    @classmethod
    def build(cls, scenarios: Sequence[str], rows: Iterable[LedgerRow]) -> ScenarioMap:
        """Строит отображение только для сценариев из шаблона.

        В реестре сотни посторонних счетов — они относятся к операциям, которых нет
        ни в одном шаблоне, и в отображение попадать не должны.
        """
        wanted = set(scenarios)
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.scenario_id in wanted:
                seen[row.scenario_id].add(row.account_id)

        missing = sorted(wanted - set(seen))
        if missing:
            raise ScenarioMappingError(f"В реестре нет операций сценариев: {', '.join(missing)}")

        # Неоднозначность не проглатываем: по условию счёт у сценария ровно один, и
        # расхождение здесь означает, что дальше поедет всё.
        ambiguous = tuple(
            (scenario, tuple(sorted(accounts)))
            for scenario, accounts in sorted(seen.items())
            if len(accounts) != 1
        )
        if ambiguous:
            details = "; ".join(f"{s}: {', '.join(a)}" for s, a in ambiguous)
            raise ScenarioMappingError(f"У сценария больше одного счёта — {details}")

        scenario_to_account = {
            scenario: next(iter(accounts)) for scenario, accounts in sorted(seen.items())
        }
        account_to_scenario = {v: k for k, v in scenario_to_account.items()}
        if len(account_to_scenario) != len(scenario_to_account):
            raise ScenarioMappingError("Один счёт достался нескольким сценариям")

        return cls(
            scenario_to_account=scenario_to_account,
            account_to_scenario=account_to_scenario,
        )
