"""Проверки прогона, не зависящие от формулировок.

Разбор документов держится на наблюдённых формулировках, и любая из них может не
совпасть на другом наборе. Шаблоны от этого не спасают: их всегда можно не угадать.
Спасает другое — проверки, опирающиеся на структуру задачи, а не на слова.

«У заёмщика ровно один действующий договор» верно независимо от того, каким словом
помечена вытесненная редакция и на каком языке. Если проверка упала, значит маркер
не распознан, — и мы узнаём об этом из отчёта, а не из потерянной ячейки.

Проверки живут в прогоне, а не в тестах: тесты гоняются на открытом наборе, а
ломается всё на приватном, где их никто не выполнит.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from halyk.ingest.inventory import DatasetInventory
from halyk.knowledge.facts import FactSet
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus

# Денежная сумма с копейками. Одиночное совпадение ничего не значит — оно попадается
# и в регламенте, и в еженедельной сводке. Значима именно их плотность.
_MONEY = re.compile(r"\$\s?[\d,]+\.\d{2}")

# Граница плотности измерена на открытом наборе: у 145 посторонних документов сумм не
# больше одной, у содержательных — три и выше, а в отчётности два-три десятка.
_FINANCIAL_SUBSTANCE = 3

# Одной плотности сумм мало. Короткое допсоглашение может нести единственный новый
# порог и при этом решать ячейку целиком, а по числу сумм не отличаться от служебной
# записки. Поэтому признаки складываются: каждый весит балл, значимым считается
# документ, набравший два.
#
# Номер счёта сам по себе в признаки не годится, хотя и напрашивается: еженедельные
# сводки и регламенты приманок его упоминают и дают двенадцать ложных срабатываний.
# В сумме с другим признаком он уже осмыслен.
_CLAUSE_NUMBER = re.compile(r"(?:Пункт|Clause|Статья|Article)\s+\d+(?:\.\d+)?")
_ACCOUNT = re.compile(r"ACC-\d{4}")
# Слова, которые обещают обязательство, а не рассказ о процессе. «KYC» и «комплаенс»
# сюда не входят намеренно: ими полны регламенты приманок, а настоящее досье и так
# опознаётся собственными маркерами типа.
_CONTRACT_WORDS = re.compile(
    r"ковенант|допсоглашени|дополнительн\w+ соглашени|отказ от прав|аудированн"
    r"|covenant|loan agreement|amendment|waiver|audited",
    re.IGNORECASE,
)
_TABLE_ROW = re.compile(r"^\|.*\|.*\|", re.M)
_SIGNIFICANT_SIGNALS = 2


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Violation:
    """Нарушенный инвариант вместе с тем, что именно не сошлось."""

    code: str
    severity: Severity
    message: str
    subject: str = ""

    def record(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "subject": self.subject,
        }


def relevance_signals(text: str) -> tuple[str, ...]:
    """Структурные признаки значимости документа.

    Возвращаются поимённо, а не одним числом: в отчёте по ним видно, чем именно
    документ похож на значимый, и спорный случай разбирается без повторного чтения.
    """
    found: list[str] = []
    if len(_MONEY.findall(text)) >= _FINANCIAL_SUBSTANCE:
        found.append("money")
    if _CLAUSE_NUMBER.search(text):
        found.append("clause_number")
    if _ACCOUNT.search(text):
        found.append("account")
    if _CONTRACT_WORDS.search(text):
        found.append("contract_vocabulary")
    if len(_TABLE_ROW.findall(text)) >= 2:
        found.append("table")
    return tuple(found)


def _looks_relevant(document: DocumentFacts) -> bool:
    """Похож ли нераспознанный документ на значимый.

    Проверка нужна ровно для случая, который мы уже пропустили однажды: тип не
    определился, документ ушёл в шум, а в нём лежала величина для ковенанта.
    """
    return len(relevance_signals(document.text)) >= _SIGNIFICANT_SIGNALS


def check_documents(inventory: DatasetInventory) -> list[Violation]:
    """Проверки переписи: что распознано, к кому отнесено и в какой редакции."""
    found: list[Violation] = []

    for document in inventory.documents:
        if document.kind.is_relevant or not _looks_relevant(document):
            continue
        # Именно так был потерян консолидированный отчёт: тип не распознан, документ
        # ушёл в шум, а вместе с ним и величина, без которой не считался ковенант.
        found.append(
            Violation(
                code="unclassified_but_relevant",
                severity=Severity.ERROR,
                message=(
                    f"документ не отнесён ни к одному типу, но содержит признаки "
                    f"значимого: {', '.join(relevance_signals(document.text))} "
                    f"(язык: {document.language_hint})"
                ),
                subject=document.file_name,
            )
        )

    for name in inventory.unresolved_documents:
        found.append(
            Violation(
                code="document_without_account",
                severity=Severity.ERROR,
                message="тип документа определён, но заёмщик не установлен",
                subject=name,
            )
        )

    found.extend(_check_editions(inventory))
    return found


def _check_editions(inventory: DatasetInventory) -> list[Violation]:
    """Ровно один действующий договор на заёмщика.

    Инвариант из условия задачи: организаторы подтвердили, что действующий договор
    всегда один. Его нарушение означает нераспознанный маркер вытеснения — и это
    самая дорогая из тихих ошибок, потому что даёт верное число при неверном пороге.
    """
    found: list[Violation] = []
    for account in sorted(set(inventory.scenarios.scenario_to_account.values())):
        agreements = [
            doc
            for doc in inventory.by_account(account)
            if doc.kind is DocumentKind.LOAN_AGREEMENT and doc.status is DocumentStatus.CURRENT
        ]
        if len(agreements) == 1:
            continue
        names = ", ".join(sorted(doc.file_name for doc in agreements)) or "нет ни одного"
        found.append(
            Violation(
                code="edition_is_ambiguous",
                severity=Severity.ERROR,
                message=(
                    f"действующих договоров должно быть ровно один, найдено "
                    f"{len(agreements)}: {names}"
                ),
                subject=account,
            )
        )
    return found


def check_facts(facts: FactSet, inventory: DatasetInventory) -> list[Violation]:
    """Проверки слоя фактов: что прочитано из документов и чего не хватило."""
    found: list[Violation] = []

    for name in facts.unparsed_dossiers:
        found.append(
            Violation(
                code="dossier_not_parsed",
                severity=Severity.ERROR,
                message="досье распознано как KYC, но политика связанных сторон не прочитана",
                subject=name,
            )
        )

    for item in facts.unparsed_disclosures:
        found.append(
            Violation(
                code="disclosure_not_parsed",
                severity=Severity.ERROR,
                message="пункт раскрытия содержит сумму или операцию, но не разобран",
                subject=item,
            )
        )

    accounts = set(inventory.scenarios.scenario_to_account.values())
    with_policy = {fact.account_id for fact in facts.facts if fact.kind == "related_party_policy"}
    for account in sorted(accounts - with_policy):
        found.append(
            Violation(
                code="related_party_policy_missing",
                severity=Severity.WARNING,
                message="у заёмщика не прочитан порог связанных сторон",
                subject=account,
            )
        )
    return found


def check_coverage(answered: Sequence[str], expected: Sequence[str]) -> list[Violation]:
    """Каждая ячейка шаблона должна получить ответ.

    Пропущенная ячейка стоит столько же, сколько неверная, поэтому её отсутствие —
    ошибка прогона, а не мелочь отчёта.
    """
    missing = sorted(set(expected) - set(answered))
    return [
        Violation(
            code="cell_not_answered",
            severity=Severity.ERROR,
            message="ячейка шаблона осталась без ответа",
            subject=cell,
        )
        for cell in missing
    ]


def summarise(violations: Sequence[Violation]) -> dict[str, object]:
    """Сводка для артефакта прогона и для кода возврата строгого режима."""
    errors = [v for v in violations if v.severity is Severity.ERROR]
    return {
        "total": len(violations),
        "errors": len(errors),
        "warnings": len(violations) - len(errors),
        "violations": [v.record() for v in violations],
    }
