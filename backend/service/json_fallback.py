"""Chat for models without tool calling: one structured-JSON decision instead of the agent router.

The model answers directly or asks for a canvas rewrite. Rewrites go through
the same path as the agent's update_workflow tool: rewrite_workflow_simple,
then apply_workflow_update, which validates before anything is saved.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel

from ..dao.workflow_table import get_workflow_data
from ..providers import ModelInfo
from ..utils.llm_settings import resolve_llm_settings
from ..utils.logger import log
from ..utils.request_context import get_rewrite_context, get_session_id
from ..utils.structured_output import complete_json
from .negotiation import to_chat_messages
from .workflow_rewrite_agent_simple import rewrite_workflow_simple
from .workflow_rewrite_tools import apply_workflow_update, node_definitions

SYSTEM_PROMPT = """You are ComfyUI-Copilot, an assistant for building and editing ComfyUI image-generation workflows.

The user's current canvas:
{canvas}

Decide what to do with the user's latest message:
- "rewrite_workflow": the user wants the current workflow changed (add/remove/replace nodes, change parameters,
  fix it) or a new workflow built on the canvas. Put a precise, self-contained description of the change in
  "rewrite_intent", naming nodes by id and inputs by their exact names shown above, and a short note for
  the user in "reply".
- "answer": anything else (questions, explanations, prompts, troubleshooting). Put the full answer in "reply",
  formatted as markdown.

Always reply in the language the user wrote in."""


REWRITE_RETRIES = 1


class FallbackDecision(BaseModel):
    action: Literal["answer", "rewrite_workflow"]
    reply: str
    rewrite_intent: str = ""


def describe_canvas(workflow: Optional[dict]) -> str:
    if not workflow:
        return "(empty canvas)"
    lines = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = []
        for name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2:
                inputs.append(f"{name}<-node {value[0]}")
            else:
                shown = json.dumps(value, ensure_ascii=False)
                inputs.append(f"{name}={shown if len(shown) <= 60 else shown[:57] + '...'}")
        lines.append(f"- node {node_id}: {node.get('class_type')} ({', '.join(inputs)})")
    return "\n".join(lines) or "(empty canvas)"


def input_names_for(workflow: Optional[dict], definitions: Optional[dict]) -> dict:
    """{node type: {"required": [...], "optional": [...]}} for the node types on the canvas."""
    if not workflow or not definitions:
        return {}
    names = {}
    for node in workflow.values():
        class_type = node.get("class_type") if isinstance(node, dict) else None
        spec = (definitions.get(class_type) or {}).get("input") if class_type else None
        if spec:
            names[class_type] = {"required": list(spec.get("required") or {}),
                                 "optional": list(spec.get("optional") or {})}
    return names


def _decide(config: dict, messages: list[dict], canvas: str) -> FallbackDecision:
    base_url, api_key, model_name = resolve_llm_settings(config)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)
    chat = [{"role": "system", "content": SYSTEM_PROMPT.format(canvas=canvas)}, *to_chat_messages(messages)]
    return complete_json(client, model_name, chat, FallbackDecision, temperature=0.2)


async def json_fallback_invoke(messages: list[dict], config: dict,
                               model: Optional[ModelInfo] = None,
                               notice: Optional[str] = None) -> AsyncGenerator[tuple[str, Any], None]:
    """Yields (text, ext) like comfyui_agent_invoke; the last item carries {"data", "finished"}."""
    session_id = get_session_id()
    workflow = get_workflow_data(session_id) if session_id else None
    prefix = f"_{notice}_\n\n" if notice else ""

    decision = await asyncio.to_thread(_decide, config, messages, describe_canvas(workflow))
    log.info(f"[json_fallback] action={decision.action} intent={decision.rewrite_intent!r}")
    text = prefix + decision.reply.strip()

    if decision.action != "rewrite_workflow":
        yield (text, {"data": None, "finished": True})
        return

    text += "\n\n▸ **Rewriting the workflow**\n\n"
    yield (text, None)

    intent = decision.rewrite_intent or decision.reply
    rewrite_context = get_rewrite_context()
    rewrite_context.current_workflow = json.dumps(workflow or {}, ensure_ascii=False)
    # Small models rename inputs (e.g. "cfg" -> "cfg_scale"); give them the real input names
    rewrite_context.node_infos = input_names_for(workflow, await node_definitions())
    result: dict = {}
    for attempt in range(1 + REWRITE_RETRIES):
        rewrite_context.rewrite_intent = intent
        workflow_json = await asyncio.to_thread(rewrite_workflow_simple, rewrite_context)
        result = await apply_workflow_update(session_id, workflow_json, "Workflow rewritten in JSON mode")
        if not result.get("error"):
            break
        log.info(f"[json_fallback] rewrite attempt {attempt + 1} rejected: {result['error']}")
        # Tell the model what was wrong (e.g. a renamed input) and let it try again
        intent = (f"{decision.rewrite_intent or decision.reply}\n\nA previous attempt was rejected: "
                  f"{result['error']} Keep every node's existing input names exactly as they are.")
    if result.get("error"):
        yield (text + f"The rewrite was not applied, so your canvas is unchanged. {result['error']}",
               {"data": None, "finished": True})
        return
    yield (text + "The workflow on the canvas has been updated.", {"data": result["ext"], "finished": True})
