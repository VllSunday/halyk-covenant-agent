import importlib
import time
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any, cast

from _pytest.monkeypatch import MonkeyPatch

from halyk.output.template import SubmissionTemplate
from halyk.pipeline.borrower import BorrowerResult
from halyk.pipeline.solve import _run_borrowers


def test_borrowers_run_in_parallel_but_return_in_template_order(
    monkeypatch: MonkeyPatch,
) -> None:
    barrier = Barrier(2, timeout=2)
    completion_order: list[str] = []
    completion_lock = Lock()

    def fake_solve(scenario: str, account: str, *_: Any, **__: Any) -> BorrowerResult:
        barrier.wait()
        if scenario == "P1":
            time.sleep(0.05)
        with completion_lock:
            completion_order.append(scenario)
        return BorrowerResult(scenario_id=scenario, account_id=account)

    solve_module = importlib.import_module("halyk.pipeline.solve")
    monkeypatch.setattr(solve_module, "solve_borrower", fake_solve)
    template = SubmissionTemplate(cells=(("P1", "6.1"), ("P2", "6.1")))
    inventory = SimpleNamespace(
        scenarios=SimpleNamespace(scenario_to_account={"P1": "A1", "P2": "A2"})
    )

    results = _run_borrowers(
        template,
        cast(Any, inventory),
        cast(Any, None),
        cast(Any, None),
        max_concurrency=2,
    )

    assert completion_order == ["P2", "P1"]
    assert [result.scenario_id for result in results] == ["P1", "P2"]
