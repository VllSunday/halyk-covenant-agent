"""Совместимость схем компилятора и resolver с Structured Outputs."""

from __future__ import annotations

from typing import Any

from halyk.compiler.contract import CompilerResponse
from halyk.llm.schema import strict_schema, unsupported
from halyk.resolution.contract import ResolverResponse


def _nodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value, *(node for child in value.values() for node in _nodes(child))]
    if isinstance(value, list):
        return [node for child in value for node in _nodes(child)]
    return []


def test_live_structured_schemas_pass_the_offline_gate() -> None:
    for model in (CompilerResponse, ResolverResponse):
        schema = strict_schema(model)
        assert unsupported(schema) == []
        assert all("pattern" not in node for node in _nodes(schema))


def test_gate_names_unsupported_regex_lookaround() -> None:
    schema = {
        "type": "object",
        "properties": {"amount": {"type": "string", "pattern": "^(?![-+.]*$).+$"}},
        "required": ["amount"],
        "additionalProperties": False,
    }

    assert unsupported(schema) == [
        "$.properties.amount.pattern: ключ не поддерживается (просмотр в регулярном выражении)"
    ]
