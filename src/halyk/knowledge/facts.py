"""Сбор фактов и корректировок по всем документам заёмщиков.

Здесь только чтение документов. Реестра этот слой не касается: сопоставление
корректировки с проводкой — отдельный шаг, и разделение нужно, чтобы «документ
велел» и «в реестре нашлось» никогда не смешивались в одну сущность.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from halyk.ingest.inventory import DatasetInventory
from halyk.knowledge import notes
from halyk.knowledge.authority import resolve_authority, superseded_drafts
from halyk.knowledge.kyc import KycError, parse_collateral_policy, parse_related_party_policy
from halyk.models.adjustment import (
    Adjustment,
    AdjustmentStatus,
    ConvertCurrencyAdjustment,
    ExcludeAdjustment,
    ReclassifyAdjustment,
    ReviewNoChangeAdjustment,
    SetEffectiveDateAdjustment,
    SetMissingAmountAdjustment,
    TransactionSelector,
)
from halyk.models.document import DocumentFacts, DocumentKind, DocumentStatus, PageFacts
from halyk.models.fact import (
    AggregateObligationFact,
    CollateralCoverageFact,
    Fact,
    FxSettlementFact,
    OneOffItemFact,
    OneOffPolicyFact,
    RelatedPartyPolicyFact,
    Share,
)
from halyk.models.source import SourceAuthority, SourceRef

_DISCLOSURE_KINDS = (
    DocumentKind.FINANCIAL_NOTES,
    DocumentKind.AUDIT_PROCEDURES,
    DocumentKind.TREASURY_MEMO,
)


@dataclass(frozen=True, slots=True)
class FactSet:
    """Всё, что документы сообщили: факты для формулы и корректировки для реестра."""

    facts: tuple[Fact, ...] = ()
    adjustments: tuple[Adjustment, ...] = ()
    unparsed_dossiers: tuple[str, ...] = field(default=())

    def by_account(self, account_id: str) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.account_id == account_id)


def _source(document: DocumentFacts, page: PageFacts, authority: SourceAuthority) -> SourceRef:
    return SourceRef(
        file_hash=document.sha256,
        file_name=document.file_name,
        page=page.number,
        kind=document.kind,
        status=document.status,
        authority=authority,
        account_id=document.account_id,
        report_number=document.report_number,
    )


def _draft_status(document: DocumentFacts, superseded: frozenset[str]) -> AdjustmentStatus:
    """Ведомость не меняет реестр никогда, но причина у этого разная.

    Заменённая окончательным отчётом и оставшаяся без него — разные ситуации: в
    первой вывод пересмотрен, во второй просто не подтверждён.
    """
    if document.file_name in superseded:
        return AdjustmentStatus.SUPERSEDED
    return AdjustmentStatus.UNCONFIRMED


def _kyc_facts(document: DocumentFacts, authority: SourceAuthority) -> list[Fact]:
    account_id = document.account_id or ""
    found: list[Fact] = []
    for page in document.pages:
        try:
            policy = parse_related_party_policy(page.text)
        except KycError:
            pass
        else:
            found.append(
                RelatedPartyPolicyFact(
                    account_id=account_id,
                    source=_source(document, page, authority),
                    threshold=policy.threshold,
                    holdings=tuple(
                        Share(counterparty=name, share=share) for name, share in policy.holdings
                    ),
                )
            )
        try:
            coverage = parse_collateral_policy(page.text)
        except KycError:
            continue
        found.append(
            CollateralCoverageFact(
                account_id=account_id,
                source=_source(document, page, authority),
                threshold=coverage.threshold,
                subsidiaries=tuple(
                    Share(counterparty=name, share=share) for name, share in coverage.coverage
                ),
            )
        )
    return found


def _page_facts(
    document: DocumentFacts, page: PageFacts, authority: SourceAuthority, item: notes.Disclosure
) -> list[Fact]:
    account_id = document.account_id or ""
    source = _source(document, page, authority)
    found: list[Fact] = []
    if (obligation := notes.aggregate_obligation(item)) is not None:
        description, amount = obligation
        found.append(
            AggregateObligationFact(
                account_id=account_id, source=source, description=description, amount=amount
            )
        )
    if (settlement := notes.fx_settlement(item)) is not None:
        counterparty, invoiced, settled = settlement
        found.append(
            FxSettlementFact(
                account_id=account_id,
                source=source,
                counterparty=counterparty,
                invoiced=invoiced,
                settled=settled,
            )
        )
    return found


def _page_adjustments(
    document: DocumentFacts,
    page: PageFacts,
    authority: SourceAuthority,
    item: notes.Disclosure,
    status: AdjustmentStatus,
) -> list[Adjustment]:
    account_id = document.account_id or ""
    source = _source(document, page, authority)
    found: list[Adjustment] = []

    common = {
        "account_id": account_id,
        "reason": item.reason or item.body,
        "source": source,
    }

    if (change := notes.reclassification(item)) is not None:
        selector = TransactionSelector(
            txn_id=change.txn_id, counterparty=change.counterparty, amount=change.amount
        )
        # Отклонённый вывод — не переклассификация с пометкой: у него нет предмета,
        # и новая категория в нём либо не названа, либо названа как отвергнутая.
        if change.accepted and change.new_value is not None:
            found.append(
                ReclassifyAdjustment(
                    selector=selector,
                    old_value=change.old_value or None,
                    new_value=change.new_value,
                    status=status,
                    **common,
                )
            )
        else:
            found.append(
                ReviewNoChangeAdjustment(
                    selector=selector,
                    old_value=change.old_value or None,
                    considered=change.new_value,
                    **common,
                )
            )
    if (missing := notes.missing_amount(item)) is not None:
        txn_id, amount = missing
        found.append(
            SetMissingAmountAdjustment(
                selector=TransactionSelector(txn_id=txn_id),
                amount=amount,
                status=status,
                **common,
            )
        )
    if (excluded := notes.excluded_transaction(item)) is not None:
        found.append(
            ExcludeAdjustment(
                selector=TransactionSelector(txn_id=excluded),
                status=status,
                **common,
            )
        )
    if (period := notes.effective_period(item)) is not None:
        txn_id, starts = period
        found.append(
            SetEffectiveDateAdjustment(
                selector=TransactionSelector(txn_id=txn_id),
                effective_date=starts,
                status=status,
                **common,
            )
        )
    return found


def _one_off_facts(
    document: DocumentFacts, page: PageFacts, authority: SourceAuthority
) -> list[Fact]:
    account_id = document.account_id or ""
    source = _source(document, page, authority)
    found: list[Fact] = [
        OneOffItemFact(
            account_id=account_id,
            source=source,
            description=description,
            counterparty=counterparty,
            amount=amount,
        )
        for description, counterparty, amount in notes.one_off_items(page.text)
    ]
    if (minimum := notes.one_off_minimum(page.text)) is not None:
        found.append(OneOffPolicyFact(account_id=account_id, source=source, minimum=minimum))
    return found


def _fx_adjustments(facts: Sequence[Fact]) -> list[Adjustment]:
    """Пересчёт валюты там, где примечания раскрыли курс по конкретному контрагенту.

    Курс не распространяется на прочие валютные операции заёмщика: примечание
    называет расчёт с одним контрагентом, и переносить его на остальные значило бы
    придумать курс, которого никто не подтверждал.
    """
    return [
        ConvertCurrencyAdjustment(
            account_id=fact.account_id,
            selector=TransactionSelector(counterparty=fact.counterparty),
            from_currency=fact.invoiced.currency,
            to_currency=fact.settled.currency,
            rate=fact.rate,
            reason=(
                f"Счёт {fact.invoiced.to_decimal()} {fact.invoiced.currency} урегулирован "
                f"платежом {fact.settled.to_decimal()} {fact.settled.currency}"
            ),
            source=fact.source,
        )
        for fact in facts
        if isinstance(fact, FxSettlementFact)
    ]


def build_facts(inventory: DatasetInventory) -> FactSet:
    """Прочитать все значимые документы и собрать факты с корректировками."""
    superseded = superseded_drafts(inventory.documents)
    facts: list[Fact] = []
    adjustments: list[Adjustment] = []
    unparsed: list[str] = []

    ordered = sorted(inventory.relevant, key=lambda doc: (doc.account_id or "", doc.file_name))
    for document in ordered:
        if not document.account_id:
            continue
        authority = resolve_authority(document)
        if authority is SourceAuthority.IGNORED:
            continue

        if document.kind is DocumentKind.KYC_FILE:
            dossier = _kyc_facts(document, authority)
            if not any(isinstance(fact, RelatedPartyPolicyFact) for fact in dossier):
                unparsed.append(document.file_name)
            facts.extend(dossier)
            continue

        if document.kind not in _DISCLOSURE_KINDS:
            continue

        status = (
            _draft_status(document, superseded)
            if document.status is DocumentStatus.DRAFT
            else AdjustmentStatus.PENDING
        )
        for page in document.pages:
            facts.extend(_one_off_facts(document, page, authority))
            for item in notes.disclosures(page.text):
                facts.extend(_page_facts(document, page, authority, item))
                adjustments.extend(_page_adjustments(document, page, authority, item, status))

    adjustments.extend(_fx_adjustments(facts))
    return FactSet(
        facts=tuple(facts),
        adjustments=tuple(adjustments),
        unparsed_dossiers=tuple(unparsed),
    )
