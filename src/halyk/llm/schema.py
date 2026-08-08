"""JSON Schema в том диалекте, который принимает строгий структурированный ответ.

Схему пишет pydantic, а провайдер принимает её подмножество: все поля обязательны,
лишние ключи запрещены, значений по умолчанию нет. Переводить схему руками нельзя —
дерево формулы рекурсивно и меняется вместе с моделями, а расхождение между схемой и
разбором обнаружилось бы только живым вызовом, то есть за деньги и в самый неудобный
момент.

Ограничения, которые провайдер не понимает, снимаются, а не переписываются: ответ
всё равно проходит через pydantic, и длина хеша проверяется там. Ослабление касается
только того, что модель увидит в подсказке, а не того, что мы примем.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# `default` строгий режим запрещает целиком: необязательных полей в нём не бывает.
# Границы длины строки в поддерживаемый набор не входят, а `discriminator` — ключ
# OpenAPI, которого в JSON Schema нет вовсе; ветки объединения и без него различаются
# по `const` внутри каждой.
_DROPPED = frozenset({"default", "discriminator", "minLength", "maxLength"})

# Ключи, за которыми лежат не схемы, а словари имён. Обходить их по общим правилам
# нельзя: поле с именем `const` превратилось бы в ограничение значения.
_NAMESPACES = ("properties", "$defs")


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Схема модели в строгой форме, пригодная для structured output."""
    hardened: dict[str, Any] = _harden(model.model_json_schema(mode="validation"))
    return hardened


def _harden(node: Any) -> Any:
    if isinstance(node, list):
        return [_harden(item) for item in node]
    if not isinstance(node, dict):
        return node

    hardened: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        hardened[key] = (
            {name: _harden(item) for name, item in value.items()}
            if key in _NAMESPACES and isinstance(value, dict)
            else _harden(value)
        )

    # `oneOf` требует, чтобы подошла ровно одна ветка, и обязывает провайдера проверять
    # остальные. Для размеченного объединения это то же самое, что `anyOf`: тег ветки
    # уникален, и совпасть может только одна.
    if (branches := hardened.pop("oneOf", None)) is not None:
        hardened["anyOf"] = branches
    if "const" in hardened:
        hardened["enum"] = [hardened.pop("const")]
    if "properties" in hardened:
        hardened["required"] = list(hardened["properties"])
        hardened["additionalProperties"] = False
    return hardened
