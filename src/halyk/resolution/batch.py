"""Дочитывание открытых требований одного заёмщика одним запросом.

Батч собирается по заёмщику, как и у компилятора, и по той же причине: запрос на
каждое требование не уложится в боевое окно. Требования в него попадают только
открытые — найденное детерминированным слоем модели не показывают вовсе, иначе она
переспорит разбор, который мы можем проверить, ответом, который проверить нечем.

Неоднозначность здесь не ошибка ответа, а его законный исход. Модели прямо велено
вернуть оба числа, если оба подходят: выбор «первого попавшегося» зависел бы от
порядка чтения документов, а требование, оставшееся неразрешённым, честно доедет до
отчёта и остановит ячейку.

Судьбу корректировки назначает код. Модель называет, что документ предписывает, а
статус выводится из права этого документа менять расчёт — иначе промежуточная
ведомость применилась бы наравне с окончательным отчётом.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic import ValidationError

from halyk.compiler.contract import Cardinality, FactRequirement
from halyk.knowledge.authority import resolve_authority
from halyk.llm.documents import index, own_documents, render, source_hashes
from halyk.llm.runner import ModelRunner, Request, Role
from halyk.llm.schema import strict_schema
from halyk.models.adjustment import (
    Adjustment,
    AdjustmentAction,
    AdjustmentStatus,
    ConvertCurrencyAdjustment,
    ExcludeAdjustment,
    IncludeAdjustment,
    ReclassifyAdjustment,
    ReviewNoChangeAdjustment,
    SetEffectiveDateAdjustment,
    SetMissingAmountAdjustment,
    TransactionSelector,
)
from halyk.models.document import DocumentFacts
from halyk.models.source import SourceAuthority, SourceRef
from halyk.money import Money
from halyk.resolution.contract import (
    INSTRUCTIONS,
    OUTPUT_CONTRACT,
    SCHEMA_NAME,
    Derivation,
    ProposedAdjustment,
    ResolvedFact,
    ResolverResponse,
    UnresolvedRequirement,
)
from halyk.resolution.validator import ResolutionError, validate


class ResolutionRejectedError(ValueError):
    """Ответ resolver не прошёл проверку. Причины — списком, с адресами."""

    def __init__(self, errors: Sequence[ResolutionError]) -> None:
        super().__init__("; ".join(f"{item.code}@{item.path}" for item in errors))
        self.errors = tuple(errors)


@dataclass(frozen=True, slots=True)
class ResolverBatch:
    """Открытые требования одного заёмщика и документы, в которых их искать."""

    account_id: str
    requirements: tuple[FactRequirement, ...]
    documents: tuple[DocumentFacts, ...]

    @classmethod
    def build(
        cls,
        account_id: str,
        requirements: Iterable[FactRequirement],
        documents: Iterable[DocumentFacts],
    ) -> ResolverBatch:
        """Собрать батч, отсеяв закрытые требования и чужие по счёту.

        Чужое требование, в отличие от чужого документа, до сборки батча доходит
        штатно: `open_requirements` собираются по всем заёмщикам сразу, и разделение
        по счетам — работа этого места.
        """
        own = sorted(
            (item for item in requirements if item.account_id == account_id and item.is_open),
            key=lambda item: item.requirement_id,
        )
        return cls(
            account_id=account_id,
            requirements=tuple(own),
            documents=own_documents(account_id, documents),
        )

    @property
    def is_empty(self) -> bool:
        return not self.requirements

    def payload(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "requirements": [_requirement(item) for item in self.requirements],
            "documents": render(self.documents),
        }


def _requirement(item: FactRequirement) -> dict[str, Any]:
    """Требование в том виде, в каком его видит модель.

    Период, область и порог существенности идут вместе с видом величины: без них
    «разовая статья» одинаково хорошо описывает и ту, что входит в EBITDA, и ту, что
    не дотянула до порога.
    """
    return {
        "requirement_id": item.requirement_id,
        "fact_kind": item.fact_kind,
        "description": item.description,
        "unit": item.unit.value,
        "currency": item.currency.value if item.currency else None,
        "scope": item.scope.value,
        "cardinality": item.cardinality.value,
        "period": {"start": item.period_start.isoformat(), "end": item.period_end.isoformat()},
        "counterparty": item.counterparty,
        "materiality_floor": (
            str(item.materiality_floor.to_decimal()) if item.materiality_floor else None
        ),
    }


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Что удалось дочитать, что оказалось спорным и что так и осталось открытым.

    Отказ модели и спор источников лежат врозь: первый она высказала сама и с
    причиной, второй вывел код из того, что подходящих величин пришло несколько.
    Слитые в один список, они выглядели бы одинаково, а чинятся по-разному.
    """

    account_id: str
    facts: tuple[ResolvedFact, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    adjustments: tuple[Adjustment, ...] = ()
    ambiguous: tuple[str, ...] = ()
    unresolved: tuple[UnresolvedRequirement, ...] = ()

    @property
    def answered(self) -> tuple[str, ...]:
        found = {fact.requirement_id for fact in self.facts}
        found |= {derivation.requirement_id for derivation in self.derivations}
        return tuple(sorted(found))

    @property
    def still_open(self) -> tuple[str, ...]:
        """Требования, которые расчёт не получит: и отказанные, и спорные."""
        refused = {item.requirement_id for item in self.unresolved}
        return tuple(sorted(refused | set(self.ambiguous)))


def _unresolved_requirements(result: ResolutionResult) -> str | None:
    if not result.still_open:
        return None
    return f"не закрыты требования: {', '.join(result.still_open)}"


def source_ref(evidence_file: str, page: int, documents: dict[str, DocumentFacts]) -> SourceRef:
    """Адрес утверждения, собранный из переписи по названному моделью файлу.

    Публичная: тем же адресом подписывается величина, которую resolver дочитал, и
    без общей сборки она пришла бы в расчёт с другим происхождением, чем
    корректировка из того же абзаца.
    """
    document = documents.get(evidence_file)
    if document is None:
        raise ValueError(f"файла {evidence_file} нет среди документов заёмщика")
    return SourceRef(
        file_hash=document.sha256,
        file_name=document.file_name,
        page=page,
        kind=document.kind,
        status=document.status,
        authority=resolve_authority(document),
        account_id=document.account_id,
        report_number=document.report_number,
    )


def _status(authority: SourceAuthority) -> AdjustmentStatus:
    """Судьба корректировки по праву её источника.

    Ведомость, которую ещё не подтвердил окончательный отчёт, реестр не меняет, но
    в отчёт попадает: «не подтверждено» и «не найдено» — разные вещи.
    """
    if authority is SourceAuthority.AUTHORITATIVE:
        return AdjustmentStatus.PENDING
    return AdjustmentStatus.UNCONFIRMED


def _selector(proposal: ProposedAdjustment) -> TransactionSelector:
    amount = (
        Money.from_decimal(proposal.selector_amount, proposal.selector_currency)
        if proposal.selector_amount is not None and proposal.selector_currency is not None
        else None
    )
    return TransactionSelector(
        txn_id=proposal.txn_id, counterparty=proposal.counterparty, amount=amount
    )


def _required[T](value: T | None, field: str) -> T:
    """Поле, обязательное для этого действия. Отсутствие — отказ, а не подстановка."""
    if value is None:
        raise ValueError(f"не названо поле {field}")
    return value


def _adjustment(  # noqa: PLR0911 — по возврату на действие, и это весь их список
    proposal: ProposedAdjustment, account_id: str, documents: dict[str, DocumentFacts]
) -> Adjustment:
    """Настоящая корректировка из предложенной. Обязательные поля задаёт действие."""
    source = source_ref(proposal.evidence.file_name, proposal.evidence.page, documents)
    common: dict[str, Any] = {
        "account_id": account_id,
        "selector": _selector(proposal),
        "reason": proposal.reason,
        "source": source,
    }
    status = _status(source.authority)

    match proposal.action:
        case AdjustmentAction.RECLASSIFY:
            return ReclassifyAdjustment(
                new_value=_required(proposal.new_value, "new_value"),
                old_value=proposal.old_value,
                status=status,
                **common,
            )
        case AdjustmentAction.SET_MISSING_AMOUNT:
            return SetMissingAmountAdjustment(amount=_money(proposal), status=status, **common)
        case AdjustmentAction.EXCLUDE:
            return ExcludeAdjustment(status=status, **common)
        case AdjustmentAction.INCLUDE:
            return IncludeAdjustment(status=status, **common)
        case AdjustmentAction.SET_EFFECTIVE_DATE:
            return SetEffectiveDateAdjustment(
                effective_date=_required(proposal.effective_date, "effective_date"),
                status=status,
                **common,
            )
        case AdjustmentAction.CONVERT_CURRENCY:
            return ConvertCurrencyAdjustment(
                rate=_required(proposal.rate, "rate"),
                from_currency=_required(proposal.from_currency, "from_currency"),
                to_currency=_required(proposal.to_currency, "to_currency"),
                status=status,
                **common,
            )
        case AdjustmentAction.REVIEW_NO_CHANGE:
            # Статус здесь задан контрактом самой корректировки: применить её нельзя
            # ни по какому праву источника, можно только записать, что вопрос решён.
            return ReviewNoChangeAdjustment(
                considered=proposal.considered,
                old_value=proposal.old_value,
                status=(
                    AdjustmentStatus.REJECTED
                    if source.authority is SourceAuthority.AUTHORITATIVE
                    else AdjustmentStatus.UNCONFIRMED
                ),
                **common,
            )


def _money(proposal: ProposedAdjustment) -> Money:
    return Money.from_decimal(
        _required(proposal.amount, "amount"), _required(proposal.currency, "currency")
    )


def _adjustments(
    proposals: Sequence[ProposedAdjustment], account_id: str, documents: dict[str, DocumentFacts]
) -> tuple[list[Adjustment], list[ResolutionError]]:
    """Собрать корректировки, отклонив те, у которых не хватает полей под действие."""
    built: list[Adjustment] = []
    errors: list[ResolutionError] = []
    for position, proposal in enumerate(proposals):
        try:
            built.append(_adjustment(proposal, account_id, documents))
        except (ValidationError, ValueError) as exc:
            errors.append(
                ResolutionError(
                    code="adjustment_is_incomplete",
                    path=f"adjustments[{position}]",
                    detail=f"{proposal.action.value}: {exc}"[:300],
                )
            )
    return built, errors


@dataclass(slots=True)
class RequirementResolver:
    """Resolver поверх единственной двери к модели.

    Собственных повторов нет: смысловой повтор делает Runner, для которого негодный
    ответ — исключение из разбора. Поэтому проверка живёт в `_parse`.
    """

    runner: ModelRunner

    def resolve(self, batch: ResolverBatch) -> ResolutionResult:
        """Дочитать открытые требования заёмщика.

        Пустой батч не идёт в модель вовсе: платить за запрос «найди ничего» незачем,
        а телеметрия попытки без вопроса только мешает читать отчёт.
        """
        if batch.is_empty:
            return ResolutionResult(account_id=batch.account_id)
        documents = index(batch.documents)
        return self.runner.run(
            self.request(batch),
            partial(self._parse, batch=batch, documents=documents),
            escalate_if=_unresolved_requirements,
        )

    def request(self, batch: ResolverBatch) -> Request:
        return Request(
            role=Role.RESOLVER,
            account_id=batch.account_id,
            instructions=INSTRUCTIONS,
            schema=strict_schema(ResolverResponse),
            schema_name=SCHEMA_NAME,
            contract=OUTPUT_CONTRACT,
            payload=batch.payload(),
            source_hashes=source_hashes(batch.documents),
        )

    @staticmethod
    def _parse(
        payload: dict[str, Any], *, batch: ResolverBatch, documents: dict[str, DocumentFacts]
    ) -> ResolutionResult:
        response = ResolverResponse.model_validate(payload)
        errors = validate(response, requirements=batch.requirements, documents=documents)
        adjustments, failed = _adjustments(response.adjustments, batch.account_id, documents)
        errors += failed
        if errors:
            raise ResolutionRejectedError(errors)

        # Покрытие входа проверено валидатором, поэтому здесь остаётся только развести
        # исходы: отказ пришёл от модели, спор вывел код.
        ambiguous = _ambiguous(response, batch.requirements)
        return ResolutionResult(
            account_id=batch.account_id,
            facts=tuple(item for item in response.facts if item.requirement_id not in ambiguous),
            derivations=tuple(
                item for item in response.derivations if item.requirement_id not in ambiguous
            ),
            adjustments=tuple(adjustments),
            ambiguous=tuple(sorted(ambiguous)),
            unresolved=tuple(
                sorted(response.unresolved_requirements, key=lambda item: item.requirement_id)
            ),
        )


def _ambiguous(
    response: ResolverResponse, requirements: Sequence[FactRequirement]
) -> frozenset[str]:
    """Требования, на которые пришло больше одного ответа при ожидаемом одном."""
    counts: defaultdict[str, int] = defaultdict(int)
    for fact in response.facts:
        counts[fact.requirement_id] += 1
    for derivation in response.derivations:
        counts[derivation.requirement_id] += 1
    return frozenset(
        item.requirement_id
        for item in requirements
        if item.cardinality is Cardinality.ONE and counts[item.requirement_id] > 1
    )
