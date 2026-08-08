"""Синтетический англоязычный датасет и подставные роли для сквозных прогонов.

Набор собирается на месте и целиком на английском: русскоязычная матрица проверяет
семантики поодиночке, а здесь важно, что они складываются в ответ. Ключа открытого
набора рядом нет — прогон обязан работать при его физическом отсутствии.

Сети в этих тестах нет ни в каком виде. Конструктор клиента подменяется взрывающимся,
поэтому любой живой вызов заканчивается падением теста, а не тихим списанием денег.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import fitz
import pytest

from halyk.compiler.batch import CovenantCompiler
from halyk.compiler.contract import (
    CompiledClause,
    CompilerResponse,
    FactRequirement,
)
from halyk.config import ModelConfig, Settings
from halyk.ingest.inventory import build_inventory
from halyk.knowledge.classifier import TransactionClassifier
from halyk.llm.cache import CacheJournal, CachePolicy, CacheRole, ModelCache
from halyk.llm.classify import CategoryClassifier, TransactionInput
from halyk.llm.runner import Budget, Request, StructuredModelRunner
from halyk.models.classification import TransactionCategory as Cat
from halyk.models.covenant import Operator, Unit
from halyk.models.formula import (
    Constant,
    CovenantFormula,
    Direction,
    FactValue,
    LedgerSum,
    RelatedParty,
    Selector,
    Sum,
)
from halyk.models.manifest import RunMode
from halyk.models.source import SourceRef
from halyk.money import Currency
from halyk.parsing.ocr import CachedOcr, OcrResponse, OpenAIVisionOcr
from halyk.pipeline.engines import Engines
from halyk.resolution.batch import RequirementResolver
from halyk.resolution.contract import Evidence, ResolvedFact, ResolverResponse
from halyk.run.context import RunContext

PERIOD = (date(2025, 1, 1), date(2025, 12, 31))
E1_ACCOUNT = "ACC-4001"
E2_ACCOUNT = "ACC-4002"
UNKNOWN_HASH = "0" * 64

# --- документы ----------------------------------------------------------------

E1_AGREEMENT = """BANK LOAN AGREEMENT
Borrower: Aktau Energy JSC
Account ACC-4001

Article 6 - Financial covenants
The measurement period runs from 2025-01-01 to 2025-12-31.

Clause 6.1 Payments to related parties. The aggregate payments of the Borrower to
related parties shall not exceed $240,000.00 during the measurement period.

Clause 6.2 Operating costs together with the restructuring obligation disclosed in
the notes shall not exceed $700,000.00 during the measurement period.

Account ACC-4001
"""

E1_SUPERSEDED = """BANK LOAN AGREEMENT
SUPERSEDED EDITION. This edition no longer applies.
Borrower: Aktau Energy JSC
Account ACC-4001

Clause 6.1 Payments to related parties shall not exceed $50,000.00.
Account ACC-4001
"""

E1_KYC = """KNOW YOUR CUSTOMER FILE
Related party review

Entity
Aktau Energy JSC
Account
ACC-4001

Share of voting rights
Sarybel Capital LLP
41.2%
Pavlodar Plant Services LLP
18.0%
Entities in which the Group holds 20.0% or more of voting rights are treated as
related parties for the purposes of the Agreement.
Account ACC-4001
"""

E1_NOTES = """NOTES TO THE FINANCIAL STATEMENTS
Account ACC-4001

(9.1) Transaction TXN-E1-0004, dated 2025-02-01, is excluded from the covenant
period. Basis: the advance relates to the following year.

(9.2) The Company recognised an aggregate obligation in respect of the restructuring
programme of $120,000.00. Basis: the programme approved by the board.

(9.3) Transaction TXN-E1-0003, originally recorded as Other expenses ($90,000.00),
has been reclassified for covenant purposes as Operating costs. Basis: the agreement.
"""

E2_AGREEMENT = """BANK LOAN AGREEMENT
Borrower: Ertis Grain LLP
Account ACC-4002

Article 6 - Financial covenants
The measurement period runs from 2025-01-01 to 2025-12-31.

Clause 6.1 Utility costs together with the guarantee obligation disclosed to the
lender shall not exceed $150,000.00 during the measurement period.

Clause 6.2 Payroll costs shall not exceed $400,000.00 during the measurement period.

Clause 6.3 EBITDA for the period shall not be less than $250,000.00.

Account ACC-4002
"""

E2_KYC = """KNOW YOUR CUSTOMER FILE
Related party review

Entity
Ertis Grain LLP
Account
ACC-4002

Share of voting rights
Semey Logistics LLP
15.0%
Entities in which the Group holds 25.0% or more of voting rights are treated as
related parties for the purposes of the Agreement.
Account ACC-4002
"""

# Обе величины названы так, что детерминированный слой их не разбирает: пункты не
# пронумерованы, формулировки не шаблонные. Ровно этот случай и достаётся resolver.
E2_GUARANTEE_QUOTE = "the amount covered by the guarantee at the reporting date is $45,000.00"
E2_EBITDA_QUOTE = "EBITDA for the period, as defined in the Agreement, amounts to $300,000.00"
E2_NOTES = f"""NOTES TO THE FINANCIAL STATEMENTS
Account ACC-4002

Note 12. Guarantees. The Company stands behind a guarantee issued in favour of the
lender; {E2_GUARANTEE_QUOTE}.

Note 13. All settlements with utility providers were made in cash during the period.

Note 14. {E2_EBITDA_QUOTE}.
Account ACC-4002
"""

# Страница без текстового слоя: одна картинка на весь лист. Нужна затем, чтобы
# распознавание в прогоне было настоящим, а не предполагаемым.
LEAFLET = "Marketing leaflet for the regional office. Opening hours and directions."

LEDGER = """txn_id,date,account_id,counterparty,description,amount,currency
TXN-E1-0001,2025-03-01,ACC-4001,Sarybel Capital L.L.P.,Advisory services,-250000.00,USD
TXN-E1-0002,2025-04-01,ACC-4001,Sarybel Capital LLP,Site servicing costs,-30000.00,USD
TXN-E1-0003,2025-05-01,ACC-4001,Northwind Catering LLP,Catering for the depot,-90000.00,USD
TXN-E1-0004,2025-02-01,ACC-4001,Sarybel Capital LLP,Advance excluded by the auditor,-500000.00,USD
TXN-E2-0001,2025-03-05,ACC-4002,City Power Grid JSC,Electricity supply,-60000.00,USD
TXN-E2-0002,2025-06-10,ACC-4002,Ertis Payroll Agency,Payroll for the second quarter,-180000.00,USD
TXN-E2-0003,2025-09-15,ACC-4002,City Water Utility LLP,Water supply and sewerage,-30000.00,USD
"""

BLANK = {"status": None, "actual": None, "evidence_txn_id": None}
TEMPLATE: dict[str, Any] = {
    "team": "",
    "contact_email": "",
    "model": "",
    "answers": {
        "E1": {"6.1": dict(BLANK), "6.2": dict(BLANK)},
        "E2": {"6.1": dict(BLANK), "6.2": dict(BLANK), "6.3": dict(BLANK)},
    },
}

CATEGORIES = {
    "TXN-E1-0001": Cat.OTHER,
    "TXN-E1-0002": Cat.OPEX,
    "TXN-E1-0003": Cat.OTHER,
    "TXN-E1-0004": Cat.OTHER,
    "TXN-E2-0001": Cat.UTILITIES,
    "TXN-E2-0002": Cat.PAYROLL,
    "TXN-E2-0003": Cat.UTILITIES,
}


def write_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    for chunk in [text[i : i + 1800] for i in range(0, len(text), 1800)] or [""]:
        page = document.new_page()
        page.insert_text((40, 60), chunk, fontsize=9)
    document.save(path)
    document.close()


def write_scan(path: Path) -> None:
    """Лист, на котором только растр: текстового слоя нет, и без OCR он пустой."""
    document = fitz.open()
    page = document.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 600, 800))
    pixmap.set_rect(pixmap.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    document.close()


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("english-e2e")
    (root / "documents").mkdir()
    for name, text in (
        ("e1-agreement.pdf", E1_AGREEMENT),
        ("e1-superseded.pdf", E1_SUPERSEDED),
        ("e1-kyc.pdf", E1_KYC),
        ("e1-notes.pdf", E1_NOTES),
        ("e2-agreement.pdf", E2_AGREEMENT),
        ("e2-kyc.pdf", E2_KYC),
        ("e2-notes.pdf", E2_NOTES),
    ):
        write_pdf(root / "documents" / name, text)
    write_scan(root / "documents" / "leaflet.pdf")
    (root / "master_ledger_2025.csv").write_text(LEDGER, encoding="utf-8")
    (root / "submission_template.json").write_text(
        json.dumps(TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


# --- ответы моделей -----------------------------------------------------------


def requirement(
    requirement_id: str,
    scenario: str,
    account: str,
    description: str,
    *,
    kind: str = "aggregate_obligation",
) -> FactRequirement:
    return FactRequirement(
        requirement_id=requirement_id,
        scenario_id=scenario,
        account_id=account,
        fact_kind=kind,
        description=description,
        unit=Unit.MONEY,
        currency=Currency.USD,
        period_start=PERIOD[0],
        period_end=PERIOD[1],
    )


def clause(
    scenario: str,
    clause_id: str,
    title: str,
    measure: Any,
    *,
    threshold: str,
    quote: str,
    file_name: str,
    operator: Operator = Operator.LE,
    requirements: tuple[FactRequirement, ...] = (),
) -> CompiledClause:
    return CompiledClause(
        formula=CovenantFormula(
            scenario_id=scenario,
            clause_id=clause_id,
            title=title,
            measure=measure,
            operator=operator,
            threshold=Constant(value=Decimal(threshold)),
            unit=Unit.MONEY,
            source_refs=(SourceRef(file_hash=UNKNOWN_HASH, file_name=file_name, page=1),),
            quote=quote,
            confidence=0.92,
        ),
        required_facts=requirements,
        period_start=PERIOD[0],
        period_end=PERIOD[1],
    )


E1_RESTRUCTURING = requirement(
    "e1-restructuring", "E1", E1_ACCOUNT, "aggregate restructuring obligation"
)
E2_GUARANTEE = requirement(
    "e2-guarantee", "E2", E2_ACCOUNT, "guarantee obligation disclosed to the lender"
)
# Вида `ebitda` среди специализированных фактов нет: величина доедет до расчёта общей.
E2_EBITDA = requirement(
    "e2-ebitda", "E2", E2_ACCOUNT, "EBITDA за период по определению договора", kind="ebitda"
)


def e1_clauses() -> tuple[CompiledClause, ...]:
    return (
        clause(
            "E1",
            "6.1",
            "Payments to related parties",
            LedgerSum(
                selector=Selector(direction=Direction.OUTFLOW, related_party=RelatedParty.ONLY)
            ),
            threshold="240000",
            quote="shall not exceed $240,000.00 during the measurement period",
            file_name="e1-agreement.pdf",
        ),
        clause(
            "E1",
            "6.2",
            "Operating costs and the restructuring obligation",
            Sum(
                terms=(
                    LedgerSum(
                        selector=Selector(categories=(Cat.OPEX,), direction=Direction.OUTFLOW)
                    ),
                    FactValue(
                        fact_kind="aggregate_obligation", description_contains="restructuring"
                    ),
                )
            ),
            threshold="700000",
            quote="shall not exceed $700,000.00 during the measurement period",
            file_name="e1-agreement.pdf",
            requirements=(E1_RESTRUCTURING,),
        ),
    )


def e2_clauses() -> tuple[CompiledClause, ...]:
    return (
        clause(
            "E2",
            "6.1",
            "Utility costs and the guarantee obligation",
            Sum(
                terms=(
                    LedgerSum(
                        selector=Selector(categories=(Cat.UTILITIES,), direction=Direction.OUTFLOW)
                    ),
                    FactValue(fact_kind="aggregate_obligation", description_contains="guarantee"),
                )
            ),
            threshold="150000",
            quote="shall not exceed $150,000.00 during the measurement period",
            file_name="e2-agreement.pdf",
            requirements=(E2_GUARANTEE,),
        ),
        clause(
            "E2",
            "6.2",
            "Payroll costs",
            LedgerSum(selector=Selector(categories=(Cat.PAYROLL,), direction=Direction.OUTFLOW)),
            threshold="400000",
            quote="Payroll costs shall not exceed $400,000.00 during the measurement period",
            file_name="e2-agreement.pdf",
        ),
        clause(
            "E2",
            "6.3",
            "EBITDA",
            FactValue(fact_kind="ebitda"),
            operator=Operator.GE,
            threshold="250000",
            quote="EBITDA for the period shall not be less than $250,000.00",
            file_name="e2-agreement.pdf",
            requirements=(E2_EBITDA,),
        ),
    )


def compiled(clauses: Iterable[CompiledClause]) -> dict[str, Any]:
    return CompilerResponse(clauses=tuple(clauses)).model_dump(mode="json")


def resolved_guarantee(amount: str = "45000") -> ResolvedFact:
    return ResolvedFact(
        requirement_id="e2-guarantee",
        fact_kind="aggregate_obligation",
        amount=Decimal(amount),
        unit=Unit.MONEY,
        currency=Currency.USD,
        evidence=(Evidence(file_name="e2-notes.pdf", page=1, quote=E2_GUARANTEE_QUOTE),),
        confidence=0.9,
    )


def resolved_ebitda(amount: str = "300000") -> ResolvedFact:
    """Величина вида, которого среди специализированных фактов нет."""
    return ResolvedFact(
        requirement_id="e2-ebitda",
        fact_kind="ebitda",
        amount=Decimal(amount),
        unit=Unit.MONEY,
        currency=Currency.USD,
        evidence=(Evidence(file_name="e2-notes.pdf", page=1, quote=E2_EBITDA_QUOTE),),
        confidence=0.85,
    )


def resolved(**parts: Any) -> dict[str, Any]:
    return ResolverResponse(**parts).model_dump(mode="json")


def compiler_answers() -> dict[str, list[dict[str, Any]]]:
    return {E1_ACCOUNT: [compiled(e1_clauses())], E2_ACCOUNT: [compiled(e2_clauses())]}


def resolver_answers() -> dict[str, list[dict[str, Any]]]:
    return {E2_ACCOUNT: [resolved(facts=(resolved_guarantee(), resolved_ebitda()))]}


# --- подставные роли ----------------------------------------------------------


class Scripted:
    """Отправка, отвечающая по счёту заёмщика заранее заготовленным ответом.

    Ответов может быть несколько: второй достаётся смысловому повтору. Счёт, для
    которого ответа не заготовлено, роняет тест — молчаливая заглушка означала бы,
    что мы проверяем не тот батч, который ушёл.
    """

    def __init__(self, answers: dict[str, list[dict[str, Any]]]) -> None:
        self.answers = {account: list(items) for account, items in answers.items()}
        self.sent: list[tuple[str, int]] = []

    def __call__(
        self, config: ModelConfig, request: Request, api_key: str, attempt: int
    ) -> tuple[dict[str, Any], tuple[int | None, int | None], str | None]:
        self.sent.append((request.account_id, attempt))
        queue = self.answers[request.account_id]
        return queue[min(attempt, len(queue)) - 1], (120, 40), None

    @property
    def accounts(self) -> list[str]:
        return [account for account, _ in self.sent]


class ScriptedClassifier(CategoryClassifier):
    """Классификатор с заранее известными ответами. До сети не доходит."""

    def _ask(
        self, rows: Sequence[TransactionInput]
    ) -> tuple[dict[str, Any], tuple[int | None, int | None, int | None], str | None]:
        items = [
            {
                "txn_id": row.txn_id,
                "category": CATEGORIES[row.txn_id].value,
                "confidence": 0.95,
                "evidence": row.description,
            }
            for row in rows
        ]
        return {"items": items}, (30, 10, 40), None


@dataclass(slots=True)
class ScriptedOcr(OpenAIVisionOcr):
    """Боевой ключ кэша без сети.

    Наследование здесь не украшение: ключ считается по имени движка и его подписи, и
    подставной движок со своим именем проверял бы кэш, которого в бою не будет.
    """

    calls: int = 0

    def recognise(self, image: bytes) -> OcrResponse:
        self.calls += 1
        return OcrResponse(text=LEAFLET, input_tokens=900, output_tokens=100, total_tokens=1000)


def prime_ocr_cache(settings: Settings, dataset: Path) -> ScriptedOcr:
    """Распознать страницы заранее — ровно то, что делает `halyk inventory --ocr`.

    Кэш общий, поэтому оплаченная здесь страница обязана стать попаданием в прогоне.
    """
    engine = ScriptedOcr(config=settings.ocr)
    build_inventory(
        dataset,
        ["E1", "E2"],
        CachedOcr(
            engine=engine,
            cache=ModelCache(
                directory=settings.cache_dir(CacheRole.OCR.value),
                policy=CachePolicy.READ_WRITE,
                role=CacheRole.OCR.value,
            ),
        ),
    )
    return engine


def shared_cache(context: RunContext, role: CacheRole, journal: CacheJournal | None = None):  # type: ignore[no-untyped-def]
    """Кэш той же формы, что в бою: общий каталог, роль и журнал прогона."""
    return ModelCache(
        directory=context.settings.cache_dir(role.value),
        policy=context.cache_policy,
        role=role.value,
        journal=journal,
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Живой вызов в этих тестах — падение, а не расход."""

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("сквозной тест попытался создать сетевого клиента")

    monkeypatch.setattr("openai.OpenAI", explode)


@pytest.fixture
def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: Path
) -> Callable[..., RunContext]:
    monkeypatch.setenv("HALYK_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("HALYK_TEAM", "halyk")
    monkeypatch.setenv("HALYK_CONTACT_EMAIL", "team@example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for name in ("HALYK_OFFLINE", "HALYK_MAX_LIVE_CALLS", "HALYK_MAX_COST_USD"):
        monkeypatch.delenv(name, raising=False)

    def build(
        *,
        mode: RunMode = RunMode.SOLVE,
        run_id: str = "e2e",
        offline: bool = False,
        fresh: bool = False,
        max_live_calls: int | None = None,
    ) -> RunContext:
        if max_live_calls is not None:
            monkeypatch.setenv("HALYK_MAX_LIVE_CALLS", str(max_live_calls))
        return RunContext.create(
            settings=Settings.from_env(offline=offline),
            input_path=dataset,
            mode=mode,
            run_id=run_id,
            fresh=fresh,
        )

    return build


@pytest.fixture
def make_engines() -> Callable[..., Engines]:
    """Роли прогона с подставной отправкой и кэшем в каталоге прогона."""

    def build(
        context: RunContext,
        *,
        compiler: dict[str, list[dict[str, Any]]] | None = None,
        resolver: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Engines:
        budget = Budget(max_live_calls=context.settings.max_live_calls)
        config = ModelConfig(name="test-model", api_key="sk-test", offline=context.settings.offline)
        journal = CacheJournal()

        def cache(role: CacheRole) -> ModelCache:
            return shared_cache(context, role, journal)

        return Engines(
            compiler=CovenantCompiler(
                runner=StructuredModelRunner(
                    config=config,
                    cache=cache(CacheRole.COMPILER),
                    budget=budget,
                    send=Scripted(compiler if compiler is not None else compiler_answers()),
                )
            ),
            resolver=RequirementResolver(
                runner=StructuredModelRunner(
                    config=config,
                    cache=cache(CacheRole.RESOLVER),
                    budget=budget,
                    send=Scripted(resolver if resolver is not None else resolver_answers()),
                )
            ),
            classifier=TransactionClassifier(
                model=ScriptedClassifier(
                    config=config, cache=cache(CacheRole.CLASSIFIER), budget=budget
                ),
                verifier=ScriptedClassifier(
                    config=config, cache=cache(CacheRole.VERIFIER), budget=budget
                ),
            ),
            budget=budget,
            ocr=CachedOcr(
                engine=ScriptedOcr(config=context.settings.ocr), cache=cache(CacheRole.OCR)
            ),
            journal=journal,
        )

    return build
