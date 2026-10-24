'''
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-06-16 16:50:17
FilePath: /comfyui_copilot/backend/service/mcp-client.py
'''
from ..service.workflow_rewrite_tools import get_current_workflow, search_node_local
from ..utils.globals import (
    DISABLE_WORKFLOW_GEN,
    LLM_DEFAULT_BASE_URL,
    MCP_CONNECT_TIMEOUT,
    REMOTE_MCP_URL,
    SEARCH_MCP_API_KEY,
    SEARCH_MCP_URL,
    get_comfyui_copilot_api_key,
)

import asyncio
import contextlib
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple

try:
    from agents.mcp import MCPServerSse
    from agents.run import Runner
    from agents import handoff, RunContextWrapper, HandoffInputData
    if not hasattr(__import__('agents'), 'Agent'):
        raise ImportError
except Exception:
    raise ImportError(
        "Detected incorrect or missing 'agents' package while loading MCP components. "
        "Please install 'openai-agents' and ensure this plugin prefers it. Commands:\n"
        "  python -m pip uninstall -y agents gym tensorflow\n"
        "  python -m pip install -U openai-agents"
    )

from ..agent_factory import create_agent, LLMConfigurationError
from ..service.workflow_rewrite_agent import create_workflow_rewrite_agent
from ..service.message_memory import message_memory_optimize
from ..service.json_fallback import json_fallback_invoke
from ..service.negotiation import chat_model_info, prepare_images, resolve_tool_mode
from ..utils.request_context import get_rewrite_context, get_session_id, get_config
from ..utils.llm_errors import describe_llm_error
from ..utils.logger import log
from openai.types.responses import ResponseTextDeltaEvent
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel

MAX_TURNS = 30
MAX_STREAM_RETRIES = 3
WORKFLOW_TOOLS = ("recall_workflow", "gen_workflow")

# Transport-level failures that are safe to retry when nothing has happened yet.
_RETRYABLE_ERROR_MARKERS = (
    "'NoneType' object has no attribute 'strip'",
    "Connection broken",
    "InvalidChunkLength",
    "socket hang up",
    "Connection reset",
)


class ImageData:
    """Image data structure to match reference implementation"""
    def __init__(self, filename: str, data: str, url: str = None):
        self.filename = filename
        self.data = data  # base64 data
        self.url = url    # uploaded URL


def is_retryable_stream_error(error: BaseException) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    message = str(error)
    return any(marker in message for marker in _RETRYABLE_ERROR_MARKERS)


def strip_trailing_whitespace_from_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for msg in msgs:
        # Only touch assistant messages to minimize impact
        if msg.get("role") != "assistant":
            cleaned.append(msg)
            continue

        msg_copy = dict(msg)
        content = msg_copy.get("content")
        if isinstance(content, str):
            msg_copy["content"] = content.rstrip()
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part = {**part, "text": part["text"].rstrip()}
                new_content.append(part)
            msg_copy["content"] = new_content
        cleaned.append(msg_copy)
    return cleaned


def build_mcp_servers(session_id: str) -> List[MCPServerSse]:
    """MCP servers configured for this install. All of them are optional."""
    servers = []
    if REMOTE_MCP_URL:
        headers = {"X-Session-Id": session_id}
        copilot_key = get_comfyui_copilot_api_key()
        if copilot_key:
            headers["Authorization"] = f"Bearer {copilot_key}"
        servers.append(MCPServerSse(
            name="copilot",
            params={"url": REMOTE_MCP_URL, "timeout": 300.0, "headers": headers},
            cache_tools_list=True,
            client_session_timeout_seconds=300.0,
        ))
    if SEARCH_MCP_URL:
        # Never forward the Copilot key to a third-party search server.
        headers = {"Authorization": f"Bearer {SEARCH_MCP_API_KEY}"} if SEARCH_MCP_API_KEY else {}
        servers.append(MCPServerSse(
            name="search",
            params={"url": SEARCH_MCP_URL, "timeout": 300.0, "headers": headers},
            cache_tools_list=True,
            client_session_timeout_seconds=300.0,
        ))
    return servers


async def connect_mcp_servers(
    stack: contextlib.AsyncExitStack,
    servers: List[MCPServerSse],
    timeout: float = MCP_CONNECT_TIMEOUT,
) -> Tuple[List[MCPServerSse], set]:
    """Connect each server independently; skip the ones that are unreachable.

    Returns the connected servers and the names of the tools they expose.
    """
    connected, tool_names = [], set()
    for server in servers:
        try:
            await asyncio.wait_for(server.connect(), timeout=timeout)
            stack.push_async_callback(server.cleanup)
            tools = await asyncio.wait_for(server.list_tools(), timeout=timeout)
        except Exception as e:
            log.warning(f"[MCP] Skipping MCP server '{server.name}': {type(e).__name__}: {e}")
            with contextlib.suppress(Exception):
                await server.cleanup()
            continue
        connected.append(server)
        tool_names.update(tool.name for tool in tools)
        log.info(f"[MCP] Connected to '{server.name}' with {len(tools)} tools")
    return connected, tool_names


def build_router_instructions(tool_names: set) -> str:
    """Router prompt that only references tools that are actually available."""
    has_recall = "recall_workflow" in tool_names
    has_gen = "gen_workflow" in tool_names and not DISABLE_WORKFLOW_GEN
    search_tools = sorted(name for name in tool_names if "search" in name and name not in WORKFLOW_TOOLS)

    if has_recall and has_gen:
        workflow_creation_instruction = """
**CASE 3: CREATE NEW / SEARCH WORKFLOW**
IF the user wants to find or generate a NEW workflow from scratch.
- Keywords: "create", "generate", "search", "find", "recommend", "生成", "查找", "推荐".
- Action: Use `recall_workflow` AND `gen_workflow`.
"""
        workflow_constraint = """
- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST ALWAYS call BOTH recall_workflow tool AND gen_workflow tool to provide comprehensive workflow options. Never call just one of these tools - both are required for complete workflow assistance. First call recall_workflow to find existing similar workflows, then call gen_workflow to generate new workflow options.
"""
    elif has_recall or has_gen:
        tool = "recall_workflow" if has_recall else "gen_workflow"
        workflow_creation_instruction = f"""
**CASE 3: SEARCH WORKFLOW**
IF the user wants to find or generate a NEW workflow.
- Keywords: "create", "generate", "search", "find", "recommend", "生成", "查找", "推荐".
- Action: Use `{tool}`.
"""
        workflow_constraint = f"""
- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST call `{tool}`.
"""
    else:
        workflow_creation_instruction = """
**CASE 3: CREATE NEW WORKFLOW**
IF the user wants a NEW workflow from scratch.
- No workflow library or generator is connected in this installation.
- Action: hand off to the `Workflow Rewrite Agent`, which builds the workflow on the current canvas from installed nodes. Mention that it starts from the current canvas.
"""
        workflow_constraint = ""

    node_search_constraint = (
        "- To answer questions about nodes (what a node does, which node to use), call `search_node_local`; it searches the nodes installed in this ComfyUI.\n"
    )
    if search_tools:
        names = ", ".join(f"`{name}`" for name in search_tools)
        node_search_constraint += (
            f"- If `search_node_local` cannot find the node or you lack information, use the web search tool(s) {names}.\n"
        )

    return f"""You are a powerful AI assistant for designing image processing workflows, capable of automating problem-solving using tools and commands.

When handing off to workflow rewrite agent or other agents, this session ID should be used for workflow data management.

### PRIMARY DIRECTIVE: INTENT CLASSIFICATION & HANDOFF
You act as a router. Your FIRST step is to classify the user's intent.

### TOOL-CALL RELIABILITY OVERRIDE (CONTEXT-TRIM SAFE)
The conversation history may be truncated for brevity and may contain ZERO tool calls/tool results.
- You MUST NOT treat "no prior tool message" as a reason to skip tool usage.
- If a CASE below requires a tool call or handoff, you MUST execute it even if you think you already know the answer.
- If a CASE below requires a tool call or handoff, your IMMEDIATE next assistant turn MUST be that tool call/handoff (do not output any natural-language explanation first).

**CASE 1: MODIFY/UPDATE/FIX CURRENT WORKFLOW (HIGHEST PRIORITY)**
IF the user wants to:
- Modify, enhance, update, or fix the CURRENT workflow/canvas.
- Add nodes/features to the CURRENT workflow (e.g., "add LoRA", "add controlnet", "fix the error").
- Change parameters in the CURRENT workflow.
- Keywords: "modify", "update", "add", "change", "fix", "current", "canvas", "修改", "更新", "添加", "画布", "加一个", "换一个", "调一下".

**ACTION:**
- You MUST IMMEDIATELY handoff to the `Workflow Rewrite Agent`.
- DO NOT call any other tools.
- DO NOT ask for more details. Just handoff.

**CASE 2: ANALYZE CURRENT WORKFLOW**
IF the user wants to:
- Analyze, explain, or understand the current workflow structure/logic.
- Ask questions about the current workflow (e.g., "how does this work?", "explain the workflow").
- Keywords: "analyze", "explain", "understand", "how it works", "workflow structure", "分析", "解释", "怎么工作的", "解读".

**ACTION:
- You MUST call `get_current_workflow` to retrieve the workflow details.
- Then, based on the returned workflow data, provide a detailed analysis or explanation to the user.

{workflow_creation_instruction}

### CONSTRAINT CHECKLIST
You must adhere to the following constraints to complete the task:

- **Tool compliance is mandatory**: If the selected CASE requires a tool/handoff, you MUST perform it. Do not answer directly without performing the required tool/handoff.
- [Important!] Respond must in the language used by the user in their question. Regardless of the language returned by the tools being called, please return the results based on the language used in the user's query.
- Ensure that the commands or tools you invoke are within the provided tool list.
- If the execution of a command or tool fails, try changing the parameters or their format before attempting again.
- Your generated responses must follow the factual information given above. Do not make up information.
- If the result obtained is incorrect, try rephrasing your approach.
- Do not query for already obtained information repeatedly. If you successfully invoked a tool and obtained relevant information, carefully confirm whether you need to invoke it again.
- Ensure that the actions you generate can be executed accurately. Actions may include specific methods and target outputs.
- When you encounter a concept, try to obtain its precise definition and analyze what inputs can yield specific values for it.
- When generating a natural language query, include all known information in the query.
- Before performing any analysis or calculation, ensure that all sub-concepts involved have been defined.
- Printing the entire content of a file is strictly prohibited, as such actions have high costs and can lead to unforeseen consequences.
- Ensure that when you call a tool, you have obtained all the input variables for that tool, and do not fabricate any input values for it.
- Respond with markdown, using a minimum of 3 heading levels (H3, H4, H5...), and when including images use the format ![alt text](url),
{workflow_constraint}
- When the user's intent is to query, return the query result directly without attempting to assist the user in performing operations.
- When the user's intent is to get prompts for image generation (like Stable Diffusion). Use specific descriptive language with proper weight modifiers (e.g., (word:1.2)), prefer English terms, and separate elements with commas. Include quality terms (high quality, detailed), style specifications (realistic, anime), lighting (cinematic, golden hour), and composition (wide shot, close up) as needed. When appropriate, include negative prompts to exclude unwanted elements. Return words divided by commas directly without any additional text.
{node_search_constraint}
- **ERROR MESSAGE ANALYSIS** - When a user pastes specific error text/logs (containing terms like "Failed", "Error", "Traceback", or stack traces), prioritize providing troubleshooting help rather than invoking search tools. Follow these steps:
  1. Analyze the error to identify the root cause (error type, affected component, missing dependencies, etc.)
  2. Explain the issue in simple terms
  3. Provide concrete, executable solutions including:
     - Specific shell commands to fix the issue (e.g., `git pull`, `pip install`, file path corrections)
     - Code snippets if applicable
     - Configuration file changes with exact paths and values
  4. If the error relates to a specific ComfyUI extension or node, include instructions for:
     - Updating the extension (`cd path/to/extension && git pull`)
     - Reinstalling dependencies
     - Alternative approaches if the extension is problematic
"""


def parse_tool_output(output: Any) -> Dict[str, Any]:
    """Normalize a tool output into {answer, data, ext, content_dict, raw_ext}."""
    output_str = str(output)
    result = {"answer": output_str, "data": None, "ext": None, "content_dict": None, "raw_ext": None}
    try:
        payload = json.loads(output_str)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(payload, dict):
        return result

    result["raw_ext"] = payload.get("ext") or None
    text = payload.get("text")
    if text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return result
        if isinstance(parsed, dict):
            result.update(answer=parsed.get("answer"), data=parsed.get("data"), ext=parsed.get("ext"))
        elif isinstance(parsed, list):
            result.update(answer=None, data=parsed)
        result["content_dict"] = parsed
    return result


def merge_workflow_results(tool_results: Dict[str, Dict[str, Any]], disable_gen: bool = DISABLE_WORKFLOW_GEN):
    """Combine recall/gen workflow tool results into (ext, finished)."""
    recall = tool_results.get("recall_workflow")
    gen = tool_results.get("gen_workflow")

    if not recall and not gen:
        for name, result in tool_results.items():
            if result.get("ext"):
                log.info(f"Using ext from {name}")
                return result["ext"], True
        return None, True

    if recall and not gen and not disable_gen:
        # Only recall_workflow was called: keep finished=False to wait for gen_workflow
        return None, False

    workflows = []
    # Generated workflows go first, then recalled library workflows
    for result in (gen, recall):
        if result and result.get("data"):
            workflows.extend(result["data"])

    seen_ids, unique_workflows = set(), []
    for workflow in workflows:
        workflow_id = workflow.get("id")
        if workflow_id and workflow_id in seen_ids:
            continue
        if workflow_id:
            seen_ids.add(workflow_id)
        unique_workflows.append(workflow)

    ext = [{"type": "workflow", "data": unique_workflows}] if unique_workflows else None
    return ext, True


async def comfyui_agent_invoke(messages: List[Dict[str, Any]], images: List[ImageData] = None):
    """
    Invoke the ComfyUI agent with optional MCP tools and image support.

    Yields:
        tuple: (text, ext) where text is accumulated text. Only the final
        yield carries ext as {"data": ext, "finished": bool}.
    """
    current_text = ""
    try:
        messages = strip_trailing_whitespace_from_messages(messages)

        session_id = get_session_id()
        config = get_config()
        if not session_id:
            raise ValueError("No session_id found in request context")
        if not config:
            raise ValueError("No config found in request context")

        # Memory compression may call an LLM synchronously; keep it off the event loop.
        log.info(f"[MCP] Original messages count: {len(messages)}")
        messages = await asyncio.to_thread(message_memory_optimize, session_id, messages)
        log.info(f"[MCP] Optimized messages count: {len(messages)}")

        # Adapt to the selected model: describe images it can't see, and skip the
        # tool-calling agent router for models without tool support
        model_info = await chat_model_info(config)
        messages = await prepare_images(messages, config, model_info)
        tool_mode = resolve_tool_mode(config, model_info)
        if tool_mode == "json":
            log.info(f"[MCP] Using JSON mode for {config.get('model_select')}")
            notice = None
            if (config.get("tool_mode") or "auto") == "auto":
                notice = (f"{model_info.id} doesn't support tool calling, so Copilot is using simple mode. "
                          "Choose a tool-capable model for the full assistant.")
            async for item in json_fallback_invoke(messages, config, model_info, notice):
                yield item
            return

        async with contextlib.AsyncExitStack() as stack:
            server_list, mcp_tool_names = await connect_mcp_servers(stack, build_mcp_servers(session_id))

            workflow_rewrite_agent_instance = create_workflow_rewrite_agent()

            class HandoffRewriteData(BaseModel):
                latest_rewrite_intent: str

            async def on_handoff(ctx: RunContextWrapper[None], input_data: HandoffRewriteData):
                get_rewrite_context().rewrite_intent = input_data.latest_rewrite_intent
                log.info(f"Rewrite agent called with intent: {input_data.latest_rewrite_intent}")

            def rewrite_handoff_input_filter(data: HandoffInputData) -> HandoffInputData:
                """Replace message history with just the rewrite intent"""
                intent = get_rewrite_context().rewrite_intent
                new_history = ()
                try:
                    for item in data.input_history:
                        if getattr(item, 'role', None) == 'user':
                            if hasattr(item, 'model_copy'):
                                new_history = (item.model_copy(update={"content": intent}),)
                                break
                            if hasattr(item, 'copy'):
                                new_history = (item.copy(update={"content": intent}),)
                                break
                except Exception as e:
                    log.warning(f"Failed to construct user message item: {e}")

                # Without a user message the agent still sees the handoff tool call, which carries the intent.
                return HandoffInputData(
                    input_history=new_history,
                    pre_handoff_items=(),
                    new_items=tuple(data.new_items),
                )

            handoff_rewrite = handoff(
                agent=workflow_rewrite_agent_instance,
                input_type=HandoffRewriteData,
                input_filter=rewrite_handoff_input_filter,
                on_handoff=on_handoff,
            )

            agent = create_agent(
                name="ComfyUI-Copilot",
                instructions=build_router_instructions(mcp_tool_names),
                mcp_servers=server_list,
                handoffs=[handoff_rewrite],
                tools=[get_current_workflow, search_node_local],
                config=config,
            )

            tool_results: Dict[str, Dict[str, Any]] = {}
            tool_call_queue: List[str] = []
            workflow_update_ext = None
            handoff_occurred = False
            # Once anything was streamed or a tool ran, restarting the run would
            # duplicate text and re-apply tool side effects, so we stop retrying.
            run_has_progress = False

            async def process_stream_events(stream_result):
                nonlocal current_text, workflow_update_ext, handoff_occurred, run_has_progress
                async for event in stream_result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        if event.data.delta:
                            current_text += event.data.delta
                            run_has_progress = True
                            yield current_text

                    elif event.type == "agent_updated_stream_event":
                        log.info(f"Handoff to: {event.new_agent.name}")
                        # Skip the initial switch to ComfyUI-Copilot itself
                        if handoff_occurred:
                            current_text += f"\n▸ **Switching to {event.new_agent.name}**\n\n"
                            yield current_text
                        handoff_occurred = True

                    elif event.type == "run_item_stream_event":
                        if event.item.type == "tool_call_item":
                            tool_name = getattr(event.item.raw_item, 'name', 'unknown_tool')
                            tool_call_queue.append(tool_name)
                            run_has_progress = True
                            log.info(f"-- Tool '{tool_name}' was called")
                        elif event.item.type == "tool_call_output_item":
                            tool_name = tool_call_queue.pop(0) if tool_call_queue else 'unknown_tool'
                            parsed = parse_tool_output(event.item.output)
                            raw_ext = parsed.pop("raw_ext")
                            if raw_ext and any(item.get("type") in ("workflow_update", "param_update") for item in raw_ext):
                                workflow_update_ext = raw_ext
                                log.info(f"-- Captured workflow ext from '{tool_name}': {len(raw_ext)} items")
                            tool_results[tool_name] = parsed

            attempt = 0
            while True:
                result = Runner.run_streamed(agent, input=messages, max_turns=MAX_TURNS)
                try:
                    async for text in process_stream_events(result):
                        yield (text, None)
                    break
                except Exception as stream_error:
                    attempt += 1
                    if run_has_progress or attempt > MAX_STREAM_RETRIES or not is_retryable_stream_error(stream_error):
                        raise
                    wait_time = min(2 ** (attempt - 1), 10)
                    log.warning(f"Stream error before any output (attempt {attempt}/{MAX_STREAM_RETRIES}), retrying in {wait_time}s: {stream_error}")
                    await asyncio.sleep(wait_time)

            log.info(f"Tool results: {list(tool_results)}")
            ext, finished = merge_workflow_results(tool_results)

            final_ext = ext
            if workflow_update_ext:
                final_ext = list(workflow_update_ext) + (ext or [])

            yield (current_text, {"data": final_ext, "finished": finished})

    except LLMConfigurationError as e:
        yield (str(e), {"data": None, "finished": True})
    except RateLimitError as e:
        body = e.body if isinstance(e.body, dict) else {}
        yield (body.get("message") or "Rate limit exceeded, please try again later.", {"data": None, "finished": True})
    except Exception as e:
        log.error(f"Error in comfyui_agent_invoke: {e}\n{traceback.format_exc()}")
        prefix = f"{current_text}\n\n" if current_text else ""
        config = get_config() or {}
        message = describe_llm_error(e, config.get("openai_base_url") or LLM_DEFAULT_BASE_URL,
                                     config.get("model_select"))
        if is_retryable_stream_error(e):
            # Keep the conversation open so the user can retry
            yield (f"{prefix}{message} Please try your request again.", {"data": None, "finished": False})
        else:
            yield (f"{prefix}An error occurred while processing your request: {message}",
                   {"data": None, "finished": True})
