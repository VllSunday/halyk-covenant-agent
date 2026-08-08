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
#
# `pattern` снимается целиком, хотя простые выражения провайдер принимает. Причина в
# том, что мы его не пишем: pydantic сам подставляет для `Decimal` шаблон с
# отрицательным просмотром вперёд, и такой диалект отвергается. Разбирать, какое
# выражение пройдёт, а какое нет, значит зависеть от чужого движка регулярных
# выражений; проверку формата всё равно делает pydantic на нашей стороне.
_DROPPED = frozenset({"default", "discriminator", "minLength", "maxLength", "pattern"})

# Эти конструкции не переписываются автоматически: их появление означает, что
# модель ответа надо выразить поддерживаемым подмножеством явно.
_FORBIDDEN = _DROPPED | frozenset(
    {
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "patternProperties",
    }
)

# Просмотры вперёд и назад: то, на чём отказ и произошёл. Проверяются отдельно, чтобы
# ошибка называла причину, а не просто «остался pattern».
LOOKAROUND = ("(?=", "(?!", "(?<=", "(?<!")

# Ключи, за которыми лежат не схемы, а словари имён. Обходить их по общим правилам
# нельзя: поле с именем `const` превратилось бы в ограничение значения.
_NAMESPACES = ("properties", "$defs")


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Схема модели в строгой форме, пригодная для structured output."""
    hardened: dict[str, Any] = _harden(model.model_json_schema(mode="validation"))
    if problems := unsupported(hardened):
        raise ValueError("Схема несовместима со structured output: " + "; ".join(problems))
    return hardened


def unsupported(schema: dict[str, Any]) -> list[str]:
    """Всё, из-за чего провайдер отвергнет схему. Пустой список — можно отправлять.

    Проверка нужна затем, чтобы несовместимость ловилась офлайн и бесплатно. Один
    раз она уже стоила живого запроса: `pattern` с просмотром вперёд приходил из
    pydantic, а узнали мы об этом ответом 400 на боевом ключе.
    """
    found: list[str] = []
    _inspect(schema, "$", found)
    if schema.get("type") != "object":
        found.append("$: корень схемы обязан быть объектом")
    return found


def _inspect(node: Any, path: str, found: list[str]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _inspect(item, f"{path}[{index}]", found)
        return
    if not isinstance(node, dict):
        return

    for key in sorted(_FORBIDDEN & node.keys()):
        value = node[key]
        marker = (
            " (просмотр в регулярном выражении)"
            if key == "pattern" and any(mark in str(value) for mark in LOOKAROUND)
            else ""
        )
        found.append(f"{path}.{key}: ключ не поддерживается{marker}")

    if "properties" in node:
        properties = node["properties"]
        if node.get("additionalProperties") is not False:
            found.append(f"{path}: additionalProperties обязан быть false")
        missing = sorted(set(properties) - set(node.get("required", [])))
        if missing:
            found.append(f"{path}.required: не перечислены поля {', '.join(missing)}")
        for name, child in properties.items():
            _inspect(child, f"{path}.properties.{name}", found)

    for key, value in node.items():
        if key in ("properties", *_FORBIDDEN):
            continue
        _inspect(value, f"{path}.{key}", found)


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
