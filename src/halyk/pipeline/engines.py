"""Всё, что ходит к моделям, собирается в одном месте.

Собрано вместе не ради удобства: у прогона один бюджет на всех, и разведённые по
стадиям объекты считали бы его порознь — каждый в своих пределах, а вместе за любыми.
По той же причине кэш у всех ролей один каталог с одной политикой: прогон либо
читает записи прошлых ответов, либо нет, и роль тут ни при чём.

Набор ролей передаётся в прогон, а не создаётся внутри него: тест собирает свой,
с подставленной отправкой, и этим доказывает, что до сети дело не дошло, — не
полагаясь на возможность пропатчить нужный метод.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halyk.compiler.batch import CovenantCompiler
from halyk.compiler.contract import INSTRUCTIONS as COMPILER_INSTRUCTIONS
from halyk.compiler.contract import OUTPUT_CONTRACT as COMPILER_CONTRACT
from halyk.compiler.contract import CompilerResponse
from halyk.hashing import sha256_payload
from halyk.knowledge.classifier import TransactionClassifier
from halyk.llm import classify
from halyk.llm.cache import CacheJournal, CacheRole, ModelCache
from halyk.llm.classify import CategoryClassifier
from halyk.llm.runner import Budget, StructuredModelRunner
from halyk.llm.schema import strict_schema
from halyk.parsing.ocr import (
    OCR_INSTRUCTIONS,
    OCR_OUTPUT_CONTRACT,
    CachedOcr,
    OpenAIVisionOcr,
)
from halyk.resolution.batch import RequirementResolver
from halyk.resolution.contract import INSTRUCTIONS as RESOLVER_INSTRUCTIONS
from halyk.resolution.contract import OUTPUT_CONTRACT as RESOLVER_CONTRACT
from halyk.resolution.contract import ResolverResponse
from halyk.run.context import RunContext


@dataclass(frozen=True, slots=True)
class Engines:
    """Четыре роли прогона, общий на них бюджет и общий журнал кэша."""

    compiler: CovenantCompiler
    resolver: RequirementResolver
    classifier: TransactionClassifier
    budget: Budget
    ocr: CachedOcr | None = None
    journal: CacheJournal = field(default_factory=CacheJournal)


def prompt_digests() -> dict[str, str]:
    """Отпечатки промптов, контрактов и схем — по одному на роль.

    Идут в манифест: правка промпта меняет ответ так же, как правка документа, и без
    отпечатка расхождение двух прогонов выглядит капризом модели.
    """
    return {
        "compiler": sha256_payload(
            {
                "instructions": COMPILER_INSTRUCTIONS,
                "contract": COMPILER_CONTRACT,
                "schema": strict_schema(CompilerResponse),
            }
        ),
        "resolver": sha256_payload(
            {
                "instructions": RESOLVER_INSTRUCTIONS,
                "contract": RESOLVER_CONTRACT,
                "schema": strict_schema(ResolverResponse),
            }
        ),
        "classifier": sha256_payload(
            {
                "instructions": classify.INSTRUCTIONS,
                "contract": classify.OUTPUT_CONTRACT,
                "schema": classify.SCHEMA,
            }
        ),
        "ocr": sha256_payload({"instructions": OCR_INSTRUCTIONS, "contract": OCR_OUTPUT_CONTRACT}),
    }


def build_engines(context: RunContext) -> Engines:
    """Роли прогона поверх общего кэша и настроек этого прогона."""
    settings = context.settings
    budget = Budget(
        max_live_calls=settings.max_live_calls,
        max_input_tokens_per_call=settings.max_input_tokens_per_call,
        max_total_input_tokens=settings.max_total_input_tokens,
        max_output_tokens=settings.max_output_tokens,
        max_estimated_cost=settings.max_cost_usd,
        price_input_per_million=settings.price_input_per_million,
        price_output_per_million=settings.price_output_per_million,
    )
    journal = CacheJournal()

    def cache(role: CacheRole) -> ModelCache:
        return ModelCache(
            directory=settings.cache_dir(role.value),
            policy=context.cache_policy,
            role=role.value,
            journal=journal,
        )

    return Engines(
        compiler=CovenantCompiler(
            runner=StructuredModelRunner(
                config=settings.compiler, cache=cache(CacheRole.COMPILER), budget=budget
            )
        ),
        # Resolver работает на настройках компилятора: он читает те же документы и
        # ошибается так же дорого. Отдельной роли в настройках нет намеренно — лишний
        # переключатель, который пришлось бы держать согласованным с компилятором.
        resolver=RequirementResolver(
            runner=StructuredModelRunner(
                config=settings.compiler, cache=cache(CacheRole.RESOLVER), budget=budget
            )
        ),
        classifier=TransactionClassifier(
            model=CategoryClassifier(
                config=settings.classifier, cache=cache(CacheRole.CLASSIFIER), budget=budget
            ),
            verifier=CategoryClassifier(
                config=settings.verifier, cache=cache(CacheRole.VERIFIER), budget=budget
            ),
        ),
        # Ключ не спрашивается заранее: страница без пригодного текстового слоя может
        # и не встретиться, а отказ за отсутствие ключа тогда стоил бы прогона.
        ocr=CachedOcr(engine=OpenAIVisionOcr(config=settings.ocr), cache=cache(CacheRole.OCR)),
        budget=budget,
        journal=journal,
    )
