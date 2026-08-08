"""Отнесение операций к статьям: документ, правило и модель как три голоса.

Порядок оснований задан не удобством, а тем, чего требуют договоры. Вывод аудитора,
применённый к операции, окончателен — «любая сумма, переклассифицированная аудиторами,
включается независимо от первоначального отражения». Дальше правило и модель
голосуют независимо: согласие принимается, расхождение уходит на проверку.

Модель спрашивают обо всех операциях, а не только о тех, где правило промолчало.
Самая опасная ошибка здесь — уверенное неверное правило: его результат никто не
перепроверит, и в сумме ковенанта окажется чужая строка. Второй независимый голос
стоит двенадцать вызовов на прогон и снимает именно этот риск.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from halyk.ingest.normalise import NormalisedLedger
from halyk.knowledge.rules import classify_by_rules
from halyk.llm.classify import CategoryClassifier, CategoryVerdict, TransactionInput
from halyk.models.adjustment import NormalisedTransaction
from halyk.models.classification import (
    ClassificationRecord,
    DecisionSource,
    TransactionCategory,
)

# Ниже этого порога решение модели само по себе не принимается. Число — сигнал, а не
# доказательство: оно не калибровано, поэтому служит только поводом спросить ещё раз.
CONFIDENCE_FLOOR = 0.75


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    records: tuple[ClassificationRecord, ...]
    usage: dict[str, Any]

    def by_id(self, txn_id: str) -> ClassificationRecord | None:
        return next((r for r in self.records if r.txn_id == txn_id), None)

    @property
    def unresolved(self) -> tuple[ClassificationRecord, ...]:
        return tuple(r for r in self.records if not r.final_category.is_decided)

    @property
    def conflicts(self) -> tuple[ClassificationRecord, ...]:
        return tuple(r for r in self.records if r.is_conflict)

    def matrix(self) -> dict[str, int]:
        counts = Counter(r.decision_source.value for r in self.records)
        return {
            "total": len(self.records),
            **{source: counts.get(source, 0) for source in (s.value for s in DecisionSource)},
            "conflicts": len(self.conflicts),
            "unknown": len(self.unresolved),
        }

    def totals(self) -> dict[str, dict[str, int]]:
        """Сколько операций в каждой статье у каждого заёмщика."""
        found: dict[str, Counter[str]] = {}
        for record in self.records:
            found.setdefault(record.account_id, Counter())[record.final_category.value] += 1
        return {account: dict(sorted(counts.items())) for account, counts in sorted(found.items())}


def _model_input(transaction: NormalisedTransaction) -> TransactionInput:
    amount = transaction.amount
    return TransactionInput(
        txn_id=transaction.txn_id,
        description=transaction.row.description,
        counterparty=transaction.row.counterparty,
        amount=str(amount.to_decimal()) if amount else "не раскрыта",
        currency=amount.currency.value if amount else "",
        date=transaction.effective_date.isoformat(),
    )


def _document_category(transaction: NormalisedTransaction) -> TransactionCategory | None:
    """Категория, названную документом, приводим к нашей онтологии.

    Документ пишет по-русски и своими словами: «Процентные расходы», «Страховые
    премии». Совпадение проверяется по началу строки, потому что формулировки в
    разных документах различаются падежом.
    """
    stated = (transaction.covenant_category or "").strip().lower()
    if not stated:
        return None
    known = {
        "процентные расходы": TransactionCategory.INTEREST_EXPENSE,
        "операционные расходы": TransactionCategory.OPEX,
        "коммунальные услуги": TransactionCategory.UTILITIES,
        "расходы на оплату труда": TransactionCategory.PAYROLL,
        "налоги": TransactionCategory.TAXES,
        "страховые премии": TransactionCategory.INSURANCE_PREMIUM,
        "капитальные затраты": TransactionCategory.CAPEX,
        "консультационные услуги": TransactionCategory.OTHER,
        "выручка": TransactionCategory.REVENUE,
    }
    for phrase, category in known.items():
        if stated.startswith(phrase):
            return category
    return None


def _from_document(
    transaction: NormalisedTransaction,
    common: dict[str, str],
    **votes: Any,
) -> ClassificationRecord:
    """Решение по операции, у которой документ назвал статью.

    Если формулировка не отображается в онтологию, строка не уходит правилам:
    молча проигнорировать вывод аудитора нельзя, он сильнее обоих наших голосов.
    """
    documented = _document_category(transaction)
    if documented is None:
        return ClassificationRecord(
            **common,
            final_category=TransactionCategory.UNKNOWN,
            decision_source=DecisionSource.UNRESOLVED,
            rule_id=votes.get("rule_id"),
            rule_category=votes.get("rule_category"),
            note=(
                f"вывод документа «{transaction.covenant_category}» "
                f"не отображается ни в одну статью"
            ),
        )
    return ClassificationRecord(
        **common,
        **votes,
        final_category=documented,
        decision_source=DecisionSource.DOCUMENT,
        note=f"вывод документа: {transaction.covenant_category}",
    )


@dataclass(slots=True)
class TransactionClassifier:
    """Три голоса и правило их разрешения."""

    model: CategoryClassifier
    verifier: CategoryClassifier | None = None
    floor: float = CONFIDENCE_FLOOR
    # Спорные строки, а не запросы: на каждого заёмщика уходит один вызов verifier
    # независимо от того, сколько строк в нём разбирается. Прежнее имя verifier_calls
    # читалось как счётчик обращений к модели и заставляло искать регресс там, где
    # его нет.
    disputed_rows: list[str] = field(default_factory=list)

    def _batches(
        self, ledger: NormalisedLedger, accounts: Sequence[str]
    ) -> dict[str, list[NormalisedTransaction]]:
        wanted = set(accounts)
        batches: dict[str, list[NormalisedTransaction]] = {
            account: [] for account in sorted(wanted)
        }
        for transaction in ledger.transactions:
            if transaction.row.account_id in wanted:
                batches[transaction.row.account_id].append(transaction)
        return batches

    def needs_check(
        self, transaction: NormalisedTransaction, verdict: CategoryVerdict | None
    ) -> bool:
        """Нужна ли этой строке независимая перепроверка.

        Проверяются спор двух голосов, неуверенность модели и молчание любого из
        них. Строку, решённую документом, проверять не у кого: выше документа
        источника нет.
        """
        if transaction.covenant_category:
            return False
        amount = transaction.amount
        match = classify_by_rules(transaction.row.description, amount.minor if amount else None)
        confident = verdict is not None and verdict.confidence >= self.floor
        return not (confident and match is not None and match.category is verdict.category)  # type: ignore[union-attr]

    def _resolve(
        self,
        account_id: str,
        transaction: NormalisedTransaction,
        verdict: CategoryVerdict | None,
        checked: CategoryVerdict | None,
    ) -> ClassificationRecord:
        common = {"txn_id": transaction.txn_id, "account_id": account_id}
        amount = transaction.amount
        match = classify_by_rules(transaction.row.description, amount.minor if amount else None)
        rule_id = match.rule_id if match else None
        rule_category = match.category if match else None
        model_category = verdict.category if verdict else None
        confidence = verdict.confidence if verdict else None
        evidence = verdict.evidence if verdict else None

        if transaction.covenant_category:
            return _from_document(
                transaction,
                common,
                rule_id=rule_id,
                rule_category=rule_category,
                model_category=model_category,
                confidence=confidence,
                evidence=evidence,
            )

        confident = verdict is not None and verdict.confidence >= self.floor
        if confident and rule_category is model_category:
            return ClassificationRecord(
                **common,
                final_category=model_category or TransactionCategory.UNKNOWN,
                decision_source=DecisionSource.AGREEMENT,
                rule_id=rule_id,
                rule_category=rule_category,
                model_category=model_category,
                confidence=confidence,
                evidence=evidence,
            )

        # Спор двух голосов, неуверенность или молчание — разбирает перепроверка.
        if checked is not None and checked.confidence >= self.floor:
            return ClassificationRecord(
                **common,
                final_category=checked.category,
                decision_source=DecisionSource.VERIFIER,
                rule_id=rule_id,
                rule_category=rule_category,
                model_category=model_category,
                confidence=checked.confidence,
                evidence=evidence,
                verifier_category=checked.category,
                verifier_reason=checked.evidence,
            )

        if confident and rule_category is None:
            return ClassificationRecord(
                **common,
                final_category=model_category or TransactionCategory.UNKNOWN,
                decision_source=DecisionSource.MODEL_ONLY,
                rule_id=None,
                model_category=model_category,
                confidence=confidence,
                evidence=evidence,
                verifier_category=checked.category if checked else None,
            )
        if rule_category is not None and verdict is None:
            return ClassificationRecord(
                **common,
                final_category=rule_category,
                decision_source=DecisionSource.RULE_ONLY,
                rule_id=rule_id,
                rule_category=rule_category,
                note="модель не ответила по этой строке",
            )

        return ClassificationRecord(
            **common,
            final_category=TransactionCategory.UNKNOWN,
            decision_source=DecisionSource.UNRESOLVED,
            rule_id=rule_id,
            rule_category=rule_category,
            model_category=model_category,
            confidence=confidence,
            evidence=evidence,
            verifier_category=checked.category if checked else None,
            note="ни один источник не дал уверенного ответа",
        )

    def run(self, ledger: NormalisedLedger, accounts: Sequence[str]) -> ClassificationResult:
        records: list[ClassificationRecord] = []
        for account_id, transactions in self._batches(ledger, accounts).items():
            verdicts = self.model.classify(account_id, [_model_input(t) for t in transactions])

            # Спорное собирается по всему заёмщику и уходит одним запросом. По
            # строке за раз это те же ответы, но вдесятеро дольше — измерено.
            doubtful = [t for t in transactions if self.needs_check(t, verdicts.get(t.txn_id))]
            self.disputed_rows.extend(t.txn_id for t in doubtful)
            checked: dict[str, CategoryVerdict] = {}
            if doubtful and self.verifier is not None:
                checked = self.verifier.classify(account_id, [_model_input(t) for t in doubtful])

            for transaction in transactions:
                records.append(
                    self._resolve(
                        account_id,
                        transaction,
                        verdicts.get(transaction.txn_id),
                        checked.get(transaction.txn_id),
                    )
                )

        usage = {
            "model": self.model.usage(),
            "verifier": self.verifier.usage() if self.verifier else None,
            "verifier_disputed_rows": len(self.disputed_rows),
        }
        return ClassificationResult(records=tuple(records), usage=usage)
