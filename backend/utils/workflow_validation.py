"""Checks for ComfyUI API-format workflows before they replace the canvas.

An LLM (especially a small local one, or a different model than the one that
started the session) can return JSON that parses but isn't a usable workflow:
nodes linked to nodes that don't exist, node types that aren't installed, or
renamed inputs (e.g. "cfg_scale" instead of KSampler's "cfg"). Saving it would
overwrite the user's canvas, so rewrites are checked first. With ComfyUI's
node definitions (/object_info) the inputs are checked the way ComfyUI's own prompt
validation does: names, link types, combo values (model files included) and number ranges.
"""

from typing import Any, Optional

from ..discovery.specs import input_config, input_options, input_type, types_compatible

# Combo options shown in an error; enough for the model to pick a real value
MAX_OPTIONS_IN_MESSAGE = 12


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
        spec = definition.get("input") or {}
        problems.extend(_input_problems(label, class_type, inputs, spec))
        problems.extend(_value_problems(label, class_type, inputs, spec, workflow, node_definitions))
    return problems


def _input_problems(label: str, class_type: str, inputs: dict, spec: dict) -> list[str]:
    required = set(spec.get("required") or {})
    known = required | set(spec.get("optional") or {}) | set(spec.get("hidden") or {})
    problems = [f"{label} ({class_type}) is missing required input '{name}'."
                for name in sorted(required - set(inputs))]
    problems += [f"{label} ({class_type}) has no input named '{name}'."
                 for name in sorted(set(inputs) - known)]
    return problems


def _value_problems(label: str, class_type: str, inputs: dict, spec: dict,
                    workflow: dict, node_definitions: dict[str, Any]) -> list[str]:
    specs = {**(spec.get("optional") or {}), **(spec.get("required") or {})}
    problems = []
    for name, value in inputs.items():
        if name not in specs:
            continue
        where = f"{label} ({class_type}) input '{name}'"
        expected = input_type(specs[name])
        if _is_link(value):
            problem = _link_problem(where, value, expected, workflow, node_definitions)
        else:
            problem = _widget_problem(where, value, expected, specs[name])
        if problem:
            problems.append(problem)
    return problems


def _link_problem(where: str, link: list, expected: Optional[str], workflow: dict,
                  node_definitions: dict[str, Any]) -> Optional[str]:
    source = workflow.get(str(link[0]))
    source_type = source.get("class_type") if isinstance(source, dict) else None
    outputs = (node_definitions.get(source_type) or {}).get("output") if isinstance(source_type, str) else None
    if not isinstance(outputs, list):
        return None  # missing source node or type: reported elsewhere
    index = link[1]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(outputs):
        return f"{where} links to output {index!r} of node {link[0]} ({source_type}), which has {len(outputs)} outputs."
    output_type = outputs[index] if isinstance(outputs[index], str) else "COMBO"
    if expected and not types_compatible(output_type, expected):
        return f"{where} expects {expected} but is linked to a {output_type} output of node {link[0]} ({source_type})."
    return None


def _widget_problem(where: str, value: Any, expected: Optional[str], spec: Any) -> Optional[str]:
    options = input_options(spec)
    if options is not None:
        if options and isinstance(value, (str, int, float)) and not isinstance(value, bool) and value not in options:
            shown = ", ".join(map(str, options[:MAX_OPTIONS_IN_MESSAGE]))
            more = f" and {len(options) - MAX_OPTIONS_IN_MESSAGE} more" if len(options) > MAX_OPTIONS_IN_MESSAGE else ""
            return f"{where} is {value!r}, which is not an available option ({shown}{more})."
        return None
    if expected not in ("INT", "FLOAT"):
        return None
    # ComfyUI converts numeric strings; anything else fails at queue time
    try:
        number = float(value) if not isinstance(value, bool) else None
    except (TypeError, ValueError):
        number = None
    if number is None:
        return f"{where} must be a number ({expected}), got {value!r}."
    config = input_config(spec)
    minimum, maximum = config.get("min"), config.get("max")
    if isinstance(minimum, (int, float)) and number < minimum:
        return f"{where} is {value!r}, below the minimum {minimum}."
    if isinstance(maximum, (int, float)) and number > maximum:
        return f"{where} is {value!r}, above the maximum {maximum}."
    return None


def _is_link(value: Any) -> bool:
    # A connection is [source_node_id, output_index]. Like ComfyUI's own prompt validation,
    # every two-element list in API format counts as one.
    return isinstance(value, list) and len(value) == 2
