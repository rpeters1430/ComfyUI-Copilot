"""Reading ComfyUI input specs from /object_info. No ComfyUI imports, so validators can use it."""

from typing import Any, Optional


def input_options(spec: Any) -> Optional[list]:
    """The allowed values of a combo input spec, in either ComfyUI format, else None.

    Legacy: [["a.safetensors", "b.safetensors"], {...}]. Current: ["COMBO", {"options": [...]}].
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        return options if isinstance(options, list) else None
    return None


def input_type(spec: Any) -> Optional[str]:
    """The socket type of an input spec ("MODEL", "INT", "COMBO", ...), or None if unknown."""
    if input_options(spec) is not None:
        return "COMBO"
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str):
        return spec[0]
    return None


def input_config(spec: Any) -> dict:
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def types_compatible(output_type: Any, expected: Any) -> bool:
    """Whether an output socket type can feed an input, by ComfyUI's rules.

    "*" matches anything, and either side may be a comma-separated set ("IMAGE,MASK")
    that matches when the sets overlap. Types that aren't plain strings aren't judged.
    """
    if not isinstance(output_type, str) or not isinstance(expected, str):
        return True
    if output_type == "*" or expected == "*" or output_type == expected:
        return True
    received = {part.strip() for part in output_type.split(",")}
    accepted = {part.strip() for part in expected.split(",")}
    return bool(received & accepted)
