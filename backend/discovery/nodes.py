"""The node types installed in this ComfyUI, from its /object_info endpoint.

/object_info is the source of truth for what an LLM may put in a workflow: class names,
input names and types, combo options (which include the installed model files) and outputs.
Building it is not free in ComfyUI (it lists model folders), so it's cached briefly.
"""

import asyncio
import time
from typing import Any, Optional

from ..utils.comfy_gateway import get_object_info
from ..utils.logger import log
from .search import query_tokens, rank
from .specs import input_config, input_options, input_type

CACHE_TTL = 30.0
FETCH_TIMEOUT = 10.0
# Combo lists can hold thousands of files; tools only need a sample to see the format
MAX_OPTIONS = 8

_cache: dict[str, Any] = {"at": 0.0, "data": None}


async def get_node_definitions(refresh: bool = False) -> Optional[dict[str, Any]]:
    """All node definitions keyed by class name, or None when ComfyUI doesn't answer."""
    if not refresh and _cache["data"] is not None and time.monotonic() - _cache["at"] < CACHE_TTL:
        return _cache["data"]
    try:
        data = await asyncio.wait_for(get_object_info(), FETCH_TIMEOUT)
    except Exception as e:
        log.warning(f"Could not load node definitions from /object_info: {e}")
        return _cache["data"]
    if not isinstance(data, dict) or not data:
        return _cache["data"]
    _cache.update(at=time.monotonic(), data=data)
    return data


def clear_node_cache() -> None:
    _cache.update(at=0.0, data=None)


def summarize_input(spec: Any, max_options: int = MAX_OPTIONS) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": input_type(spec) or "unknown"}
    options = input_options(spec)
    if options is not None:
        summary["options"] = options[:max_options]
        if len(options) > max_options:
            summary["options_total"] = len(options)
    config = input_config(spec)
    for key in ("default", "min", "max", "step", "multiline", "tooltip"):
        if key in config and not (key == "default" and isinstance(config[key], list)):
            summary[key] = config[key]
    return summary


def summarize_node(class_name: str, definition: dict[str, Any], max_options: int = MAX_OPTIONS) -> dict[str, Any]:
    """A compact node spec: what an LLM needs to wire the node correctly, without the bulk."""
    spec = definition.get("input") or {}
    inputs = {}
    for section in ("required", "optional"):
        for name, input_spec in (spec.get(section) or {}).items():
            inputs[name] = {**summarize_input(input_spec, max_options), "required": section == "required"}
    output_types = definition.get("output") or []
    output_names = definition.get("output_name") or []
    outputs = [{"index": index,
                "name": output_names[index] if index < len(output_names) else str(output_type),
                "type": output_type if isinstance(output_type, str) else "COMBO"}
               for index, output_type in enumerate(output_types)]
    return {
        "class_name": class_name,
        "display_name": definition.get("display_name") or class_name,
        "category": definition.get("category") or "",
        "description": (definition.get("description") or "")[:300],
        "python_module": definition.get("python_module") or "",
        "inputs": inputs,
        "outputs": outputs,
    }


def node_pack(definition: dict[str, Any]) -> str:
    """The custom node pack a node comes from ("comfyui-impact-pack"), or "core"."""
    module = str(definition.get("python_module") or "")
    if module.startswith("custom_nodes."):
        return module.split(".", 2)[1]
    return "core" if module else ""


def _search_fields(class_name: str, definition: dict[str, Any]):
    spec = definition.get("input") or {}
    input_names = [name for section in ("required", "optional") for name in (spec.get(section) or {})]
    return [
        (class_name, 10),
        (definition.get("display_name"), 8),
        (definition.get("category"), 4),
        (" ".join(map(str, definition.get("output_name") or [])), 3),
        (" ".join(input_names), 3),
        (node_pack(definition), 2),
        (definition.get("description"), 1),
    ]


def search_nodes(definitions: dict[str, Any], query: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Installed nodes matching `query` (a string or keyword list), best first."""
    tokens = query_tokens(query)
    if not tokens:
        return []
    ranked = rank(definitions.items(), tokens, lambda item: _search_fields(*item), limit)
    return [{
        "class_name": class_name,
        "display_name": definition.get("display_name") or class_name,
        "category": definition.get("category") or "",
        "pack": node_pack(definition),
        "score": round(points, 1),
    } for points, (class_name, definition) in ranked]


def find_class(definitions: dict[str, Any], name: str) -> Optional[str]:
    """The installed class for `name`, matching case and spacing loosely ("ksampler" -> "KSampler")."""
    if name in definitions:
        return name
    wanted = "".join(name.lower().split())
    for class_name, definition in definitions.items():
        if wanted in ("".join(class_name.lower().split()),
                      "".join(str(definition.get("display_name") or "").lower().split())):
            return class_name
    return None


def missing_node_types(node_types: set[str], definitions: Optional[dict[str, Any]]) -> list[str]:
    if definitions is None:
        return []
    return sorted(node_type for node_type in node_types if node_type not in definitions)
