"""Детерминированные правила отнесения операции к статье.

Правило — это пара «шаблон описания + направление операции», а не одно ключевое
слово. В реестре рядом лежат «quarterly interest coupon» и «interest income on
treasury bills», «property insurance premium» и «insurance broker rebate»: слово одно,
статьи разные, и различает их знак суммы. Правило без направления уверенно относило бы
поступление к расходной статье — а уверенная ошибка правила опаснее отказа модели,
потому что её никто не перепроверяет.

Правила здесь — не единственный источник решения. Они работают вторым голосом рядом с
моделью, и расхождение между ними отправляется на проверку, а не разрешается молча.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from halyk.models.classification import TransactionCategory


class Direction(StrEnum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    ANY = "any"

    @classmethod
    def of(cls, minor: int | None) -> Direction:
        if minor is None:
            return Direction.ANY
        return Direction.INFLOW if minor > 0 else Direction.OUTFLOW

    def accepts(self, other: Direction) -> bool:
        return self is Direction.ANY or other is Direction.ANY or self is other


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    category: TransactionCategory
    pattern: re.Pattern[str]
    direction: Direction = Direction.ANY


def _rule(
    rule_id: str, category: TransactionCategory, pattern: str, direction: Direction = Direction.ANY
) -> Rule:
    return Rule(rule_id, category, re.compile(pattern, re.IGNORECASE), direction)


# Возвраты, скидки и зачёты. Проверяются первыми: без них «rent overpayment refunded»
# попадёт в аренду, «excise tax credit received» — в налоги, и обе статьи вырастут на
# сумму, которую заёмщик получил, а не потратил.
_CONTRA = (
    r"(refund|rebate|returned|credit received|reimbursement|recovery|recovered"
    r"|incentive received|reversal|overfunding|adjustment credit|service credit)"
)

RULES: tuple[Rule, ...] = (
    # Поступления, названные возвратом по конкретной статье.
    _rule("contra-rent", TransactionCategory.RENT, rf"(rent|lease).*{_CONTRA}", Direction.INFLOW),
    _rule("contra-tax", TransactionCategory.TAXES, rf"tax.*{_CONTRA}", Direction.INFLOW),
    _rule(
        "contra-insurance",
        TransactionCategory.INSURANCE_PREMIUM,
        rf"insur.*{_CONTRA}",
        Direction.INFLOW,
    ),
    _rule("contra-payroll", TransactionCategory.PAYROLL, rf"payroll.*{_CONTRA}", Direction.INFLOW),
    _rule(
        "contra-utility",
        TransactionCategory.UTILITIES,
        rf"(utility|electric|water|gas|telecom).*{_CONTRA}",
        Direction.INFLOW,
    ),
    _rule(
        "contra-marketing",
        TransactionCategory.OPEX,
        rf"(marketing|ad campaign|media).*{_CONTRA}",
        Direction.INFLOW,
    ),
    # Проценты: полученные — это не процентный расход.
    _rule(
        "interest-income",
        TransactionCategory.OTHER,
        r"interest (income|credited|recovery)",
        Direction.INFLOW,
    ),
    _rule(
        "interest-expense",
        TransactionCategory.INTEREST_EXPENSE,
        r"\b(interest|coupon)\b",
        Direction.OUTFLOW,
    ),
    # Финансирование: привлечение средств, а не выручка.
    _rule(
        "financing",
        TransactionCategory.FINANCING_INFLOW,
        r"(loan (drawdown|proceeds|advance)|facility drawdown|bridge (loan|facility)"
        r"|note issue|financing proceeds|credit line drawdown|subordinated (loan|notes) received)",
        Direction.INFLOW,
    ),
    # Расходные статьи.
    _rule(
        "tax",
        TransactionCategory.TAXES,
        r"\b(tax|duty|levy|customs|vat)\b|excise|withholding",
        Direction.OUTFLOW,
    ),
    _rule(
        "payroll",
        TransactionCategory.PAYROLL,
        r"payroll|salar|wage|staff (cost|payment)|personnel|severance|pension|bonus",
        Direction.OUTFLOW,
    ),
    _rule(
        "utility",
        TransactionCategory.UTILITIES,
        r"electric|water (charge|supply)|natural gas|utility|telecom|heating|"
        r"power (supply|network)|metering",
        Direction.OUTFLOW,
    ),
    # «site hire» в правило не входит: в реестре так названа аренда рекламного места
    # («outdoor marketing site hire»), и это расходы на маркетинг, а не на помещения.
    # Правило по этим двум словам уводило в аренду семь миллионов и ломало P10/6.1.
    _rule(
        "rent",
        TransactionCategory.RENT,
        r"\b(rent|lease|tenancy)\b",
        Direction.OUTFLOW,
    ),
    _rule(
        "insurance",
        TransactionCategory.INSURANCE_PREMIUM,
        r"insurance (premium|policy)|underwriting|fidelity bond|risk survey",
        Direction.OUTFLOW,
    ),
    _rule(
        "capex",
        TransactionCategory.CAPEX,
        r"capital (expenditure|asset|work)|capitalised|equipment purchase|"
        r"construction|machinery|plant acquisition|fit-out|transfer of .*equipment",
        Direction.OUTFLOW,
    ),
    _rule(
        "revenue",
        TransactionCategory.REVENUE,
        r"sales (settlement|receipt|proceeds)|revenue|customer receipt|"
        r"service income|freight income|throughput",
        Direction.INFLOW,
    ),
    # Всё остальное операционное. Идёт последним: слово «servicing» встречается и в
    # ремонте оборудования, и в обслуживании кредита.
    _rule(
        "opex",
        TransactionCategory.OPEX,
        r"marketing|ad campaign|advertis|sponsor|exhibition|brand|newsletter|media buy|"
        r"advisory|consult|retainer|audit fee|legal|servicing|maintenance|operating costs|"
        r"cleaning|inspection|survey",
        Direction.OUTFLOW,
    ),
)


@dataclass(frozen=True, slots=True)
class RuleMatch:
    rule_id: str
    category: TransactionCategory


def classify_by_rules(description: str, minor: int | None) -> RuleMatch | None:
    """Первое подошедшее правило. Порядок в таблице значим и задан явно."""
    direction = Direction.of(minor)
    for rule in RULES:
        if rule.direction.accepts(direction) and rule.pattern.search(description):
            return RuleMatch(rule_id=rule.id, category=rule.category)
    return None
