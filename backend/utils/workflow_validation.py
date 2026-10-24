"""Checks for ComfyUI API-format workflows before they replace the canvas.

An LLM (especially a small local one, or a different model than the one that
started the session) can return JSON that parses but isn't a usable workflow:
nodes linked to nodes that don't exist, node types that aren't installed, or
renamed inputs (e.g. "cfg_scale" instead of KSampler's "cfg"). Saving it would
overwrite the user's canvas, so rewrites are checked first. With ComfyUI's
node definitions (/object_info) the inputs are checked the way ComfyUI does.
"""

from typing import Any, Optional


def validate_api_workflow(workflow: Any, node_definitions: Optional[dict[str, Any]] = None) -> list[str]:
    """Return a list of problems; an empty list means the workflow can be applied."""
    if not isinstance(workflow, dict) or not workflow:
        return ["The workflow must be a non-empty JSON object of nodes keyed by node id."]

    problems = []
    for node_id, node in workflow.items():
        label = f"Node {node_id}"
        if not isinstance(node, dict):
            problems.append(f"{label} is not an object.")
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if not isinstance(class_type, str) or not class_type:
            problems.append(f"{label} has no class_type.")
            continue
        if not isinstance(inputs, dict):
            problems.append(f"{label} has inputs that are not an object.")
            continue

        for name, value in inputs.items():
            if _is_link(value) and str(value[0]) not in workflow:
                problems.append(f"{label} input '{name}' links to node {value[0]}, which does not exist.")

        if node_definitions is None:
            continue
        definition = node_definitions.get(class_type)
        if definition is None:
            problems.append(f"{label} uses node type '{class_type}', which is not installed.")
            continue
        problems.extend(_input_problems(label, class_type, inputs, definition.get("input") or {}))
    return problems


def _input_problems(label: str, class_type: str, inputs: dict, spec: dict) -> list[str]:
    required = set(spec.get("required") or {})
    known = required | set(spec.get("optional") or {}) | set(spec.get("hidden") or {})
    problems = [f"{label} ({class_type}) is missing required input '{name}'."
                for name in sorted(required - set(inputs))]
    problems += [f"{label} ({class_type}) has no input named '{name}'."
                 for name in sorted(set(inputs) - known)]
    return problems


def _is_link(value: Any) -> bool:
    # A connection is [source_node_id, output_index]. Like ComfyUI's own prompt validation,
    # every two-element list in API format counts as one.
    return isinstance(value, list) and len(value) == 2
