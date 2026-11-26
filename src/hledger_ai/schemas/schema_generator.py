"""Generate JSON Schemas from hledger-core dataclasses.

Each extraction module declares which field paths it cares about
(e.g. ``["shop_identifier.name", "shop_identifier.address"]``).
This module walks the dataclass tree and produces a JSON Schema
containing only those fields, suitable for Ollama structured output.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    get_args,
    get_origin,
    get_type_hints,
)

from typeguard import typechecked

# Fields that no extraction module should ever try to extract from an
# image — they are runtime objects, computed, or user-entered in TUI.
GLOBAL_SKIP_FIELDS: Set[str] = {
    "config",
    "transaction_hash",
    "account_transactions",
    "original_transaction",
    "ai_receipt_categorisation",
    "withdrawal_metadata",
    "payed_for_item_rounded",
    "parent_receipt_category",
    "tags",  # events/projects and traceability, typed by the user
}


def _safe_get_type_hints(cls: type) -> Dict[str, Any]:
    """Resolve type hints, importing missing names as needed.

    ``get_type_hints`` fails when forward-referenced types
    (e.g. ``Config``) are not importable in the class's module
    globals — typically because they're behind ``TYPE_CHECKING``.
    We build a namespace with those missing names and retry.
    """
    try:
        return get_type_hints(cls)
    except NameError:
        pass

    # Build a namespace from the class's own module
    mod = sys.modules.get(cls.__module__, None)
    ns: Dict[str, Any] = {}
    if mod is not None:
        ns.update(vars(mod))

    # Resolve TYPE_CHECKING-only imports that are missing.
    # Walk the raw annotations to find unresolved names, then
    # try importing them from known packages.
    _resolve_missing_names(cls, ns)

    try:
        return get_type_hints(cls, globalns=ns)
    except Exception:
        # Last resort: return raw annotations (strings).
        return {
            f.name: f.type
            for f in dataclasses.fields(cls)
            if f.type is not dataclasses.MISSING
        }


def _resolve_missing_names(
    cls: type, ns: Dict[str, Any]
) -> None:
    """Try to import names that are referenced in annotations
    but missing from the namespace."""
    # Collect all top-level names used in annotations
    raw_annotations = getattr(cls, "__annotations__", {})
    for ann in raw_annotations.values():
        if isinstance(ann, str):
            # Extract bare names (before [ or after , etc.)
            for token in ann.replace("[", " ").replace(
                "]", " "
            ).replace(",", " ").split():
                if token and token[0].isupper() and token not in ns:
                    _try_import_name(token, ns)


def _try_import_name(
    name: str, ns: Dict[str, Any]
) -> None:
    """Try to find and import a class by name from known packages."""
    search_packages = [
        f"hledger_config.config.{name}",
        f"hledger_core.TransactionObjects.{name}",
        f"hledger_core.generics.{name}",
    ]
    for fqn in search_packages:
        try:
            __import__(fqn)
            mod = sys.modules.get(fqn)
            if mod is not None:
                cls_obj = getattr(mod, name, None)
                if cls_obj is not None and isinstance(
                    cls_obj, type
                ):
                    ns[name] = cls_obj
                    return
        except ImportError:
            continue


def _python_type_to_json_schema(tp: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    # Unresolved forward reference (string annotation)
    if isinstance(tp, str):
        return {"type": "string"}

    origin = get_origin(tp)
    args = get_args(tp)

    # NoneType
    if tp is type(None):
        return {"type": "null"}

    # Union — check for Optional pattern
    if _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X]
            inner = _python_type_to_json_schema(non_none[0])
            return {**inner, "nullable": True}
        # True union — anyOf
        return {
            "anyOf": [
                _python_type_to_json_schema(a)
                for a in args
            ]
        }

    if origin is list or origin is List:
        if args:
            return {
                "type": "array",
                "items": _python_type_to_json_schema(args[0]),
            }
        return {"type": "array"}

    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is datetime:
        return {"type": "string", "format": "date-time"}

    # Dataclass → nested object
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _dataclass_to_schema(
            tp, skip_fields=GLOBAL_SKIP_FIELDS
        )

    # Enum
    if hasattr(tp, "__members__"):
        return {
            "type": "string",
            "enum": [m.value for m in tp],
        }

    # Fallback
    return {"type": "string"}


def _is_union(origin: Any) -> bool:
    """Check if an origin type is a Union."""
    if sys.version_info >= (3, 10):
        import types

        if origin is types.UnionType:
            return True
    try:
        import typing

        if origin is typing.Union:
            return True
    except AttributeError:
        pass
    return False


def _dataclass_to_schema(
    cls: type,
    *,
    skip_fields: Set[str],
    include_only: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert a dataclass to a JSON Schema object.

    If *include_only* is set, only those top-level field names are
    included.  Nested dataclass fields are always fully expanded.
    """
    if not dataclasses.is_dataclass(cls):
        return {"type": "object"}

    hints = _safe_get_type_hints(cls)
    fields = dataclasses.fields(cls)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for field in fields:
        if field.name in skip_fields:
            continue
        if (
            include_only is not None
            and field.name not in include_only
        ):
            continue

        tp = hints.get(field.name, str)
        prop_schema = _python_type_to_json_schema(tp)
        properties[field.name] = prop_schema

        # A field is required only if it has no default and is
        # not Optional.
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        is_optional = "nullable" in prop_schema
        if not has_default and not is_optional:
            required.append(field.name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


@typechecked
def generate_extraction_schema(
    *,
    source_class: type,
    fields: List[str],
) -> Dict[str, Any]:
    """Build a JSON Schema for a subset of a dataclass's fields.

    *fields* is a list of dot-separated paths, e.g.::

        ["shop_identifier.name", "shop_identifier.address"]

    The returned schema contains only the requested paths, with
    nested dataclasses expanded into nested JSON objects.
    """
    top_level_names: Set[str] = set()
    nested_paths: Dict[str, Set[str]] = {}

    for path in fields:
        parts = path.split(".", maxsplit=1)
        top_name = parts[0]
        top_level_names.add(top_name)
        if len(parts) > 1:
            nested_paths.setdefault(top_name, set()).add(
                parts[1]
            )

    schema = _dataclass_to_schema(
        source_class,
        skip_fields=GLOBAL_SKIP_FIELDS,
        include_only=top_level_names,
    )

    # For nested paths, trim the child schema to only include
    # the requested sub-fields.
    hints = _safe_get_type_hints(source_class)
    for top_name, sub_fields in nested_paths.items():
        tp = hints.get(top_name)
        if tp is None:
            continue
        # Unwrap Optional
        args = get_args(tp)
        if args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                tp = non_none[0]

        if dataclasses.is_dataclass(tp) and isinstance(
            tp, type
        ):
            child_top = {
                p.split(".")[0] for p in sub_fields
            }
            child_schema = _dataclass_to_schema(
                tp,
                skip_fields=GLOBAL_SKIP_FIELDS,
                include_only=child_top,
            )
            schema["properties"][top_name] = child_schema

    return schema


@typechecked
def schema_version_hash(*, schema: Dict[str, Any]) -> str:
    """Deterministic hash of a schema for versioning."""
    canonical = json.dumps(schema, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@typechecked
def build_extraction_prompt(
    *, schema: Dict[str, Any], context: str = ""
) -> str:
    """Generate a VLM prompt from a JSON Schema.

    Lists the top-level property names so the model knows what to
    look for, without hardcoding field names.
    """
    props = schema.get("properties", {})
    field_names = sorted(props.keys())
    base = (
        "Extract structured data from this receipt/invoice image. "
        f"Fields to extract: {', '.join(field_names)}. "
        "If a field is not visible, set it to null."
    )
    if context:
        base = f"{context}\n\n{base}"
    return base
