# Workflow Rewrite Agent for ComfyUI Workflow Structure Fixes

import asyncio
import json
import time
from typing import Dict, Any, Optional

try:
    from agents import RunContextWrapper
    from agents.tool import function_tool
    if not hasattr(__import__('agents'), 'Agent'):
        # Ensure we are not importing RL agents
        raise ImportError
except Exception:
    raise ImportError(
        "Detected incorrect or missing 'agents' package while loading tools. "
        "Please install 'openai-agents' and ensure this plugin prefers it. Commands:\n"
        "  python -m pip uninstall -y agents gym tensorflow\n"
        "  python -m pip install -U openai-agents\n"
    )
from .workflow_rewrite_agent_simple import rewrite_workflow_simple

from ..dao.workflow_table import get_workflow_data, save_workflow_data, get_workflow_data_ui, get_workflow_data_by_id
from ..discovery.nodes import find_class, get_node_definitions, search_nodes, summarize_node
from ..utils.request_context import get_rewrite_context, get_session_id
from ..utils.logger import log
from ..utils.workflow_validation import validate_api_workflow

def get_workflow_data_from_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取工作流数据，优先使用checkpoint_id，如果没有则使用session_id"""
    workflow_checkpoint_id = config.get('workflow_checkpoint_id')
    session_id = config.get('session_id')
    
    if workflow_checkpoint_id:
        try:
            checkpoint_data = get_workflow_data_by_id(workflow_checkpoint_id)
            if checkpoint_data and checkpoint_data.get('workflow_data'):
                return checkpoint_data['workflow_data']
        except Exception as e:
            log.error(f"Failed to get workflow data from checkpoint {workflow_checkpoint_id}: {str(e)}")
    
    if session_id:
        return get_workflow_data(session_id)
    
    return None

def get_workflow_data_ui_from_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取工作流UI数据，优先使用checkpoint_id，如果没有则使用session_id"""
    workflow_checkpoint_id = config.get('workflow_checkpoint_id')
    session_id = config.get('session_id')
    
    if workflow_checkpoint_id:
        try:
            checkpoint_data = get_workflow_data_by_id(workflow_checkpoint_id)
            if checkpoint_data and checkpoint_data.get('workflow_data_ui'):
                return checkpoint_data['workflow_data_ui']
        except Exception as e:
            log.error(f"Failed to get workflow UI data from checkpoint {workflow_checkpoint_id}: {str(e)}")
    
    if session_id:
        return get_workflow_data_ui(session_id)
    
    return None

@function_tool
def get_current_workflow() -> str:
    """获取当前session的工作流数据"""
    session_id = get_session_id()
    if not session_id:
        return json.dumps({"error": "No session_id found in context"})
    
    workflow_data = get_workflow_data(session_id)
    if not workflow_data:
        return json.dumps({"error": "No workflow data found for this session"})
    
    workflow_data_str = json.dumps(workflow_data, ensure_ascii=False)
    get_rewrite_context().current_workflow = workflow_data_str
    return workflow_data_str

def _not_installed(node_class: str, definitions: Dict[str, Any]) -> Dict[str, Any]:
    suggestions = [hit["class_name"] for hit in search_nodes(definitions, node_class, limit=5)]
    return {"error": f"Node class '{node_class}' is not installed in this ComfyUI",
            "suggestions": suggestions}


@function_tool
async def get_node_info(node_class: str) -> str:
    """获取节点的详细信息，包括输入输出参数"""
    definitions = await get_node_definitions()
    if not definitions:
        return json.dumps({"error": "Could not load node definitions from ComfyUI"})
    class_name = find_class(definitions, node_class)
    if class_name is None:
        return json.dumps(_not_installed(node_class, definitions), ensure_ascii=False)
    node_info_str = json.dumps({class_name: definitions[class_name]}, ensure_ascii=False)
    get_rewrite_context().node_infos[class_name] = node_info_str
    return node_info_str


@function_tool
async def get_node_infos(node_class_list: list[str]) -> str:
    """Get the specs of up to 5 installed nodes: every input (type, required, options, default,
    min/max) and every output (index, name, type). Use exact class names from search_node_local."""
    definitions = await get_node_definitions()
    if not definitions:
        return json.dumps({"error": "Could not load node definitions from ComfyUI"})
    node_infos = {}
    for node_class in node_class_list[:10]:
        class_name = find_class(definitions, node_class)
        if class_name is None:
            node_infos[node_class] = _not_installed(node_class, definitions)
            continue
        node_infos[class_name] = summarize_node(class_name, definitions[class_name])
        get_rewrite_context().node_infos[class_name] = json.dumps(node_infos[class_name], ensure_ascii=False)
    return json.dumps(node_infos, ensure_ascii=False)


@function_tool
async def search_node_local(node_class: str = "", keywords: list[str] = None, limit: int = 10) -> str:
    """
    Search the nodes installed in this ComfyUI. Only nodes returned here (or by get_node_infos)
    exist; never invent node class names or input names.
    - node_class: a class name you already know ("KSampler", "LoraLoader"). An exact or
      case-insensitive match returns that node's full spec.
    - keywords: 1-3 specific English words for what the node does or its inputs
      ("upscale", "controlnet", "brightness"). Avoid broad words like "image".
    Returns ranked candidates with class_name, display_name, category and pack.
    """
    definitions = await get_node_definitions()
    if not definitions:
        return json.dumps({"error": "Could not load node definitions from ComfyUI"})

    node_class_str = (node_class or "").strip()
    keyword_list = [str(kw).strip() for kw in (keywords or []) if str(kw or "").strip()]
    if not node_class_str and not keyword_list:
        return json.dumps({"error": "both node_class and keywords are empty"})

    if node_class_str:
        class_name = find_class(definitions, node_class_str)
        if class_name is not None:
            summary = summarize_node(class_name, definitions[class_name])
            get_rewrite_context().node_infos[class_name] = json.dumps(summary, ensure_ascii=False)
            return json.dumps({"node_class": node_class_str, "keywords": keyword_list,
                               "match_type": "exact_class", "results": {class_name: summary}},
                              ensure_ascii=False)

    limit = max(1, min(int(limit or 10), 50))
    results = search_nodes(definitions, [node_class_str, *keyword_list], limit)
    response = {"node_class": node_class_str, "keywords": keyword_list, "match_type": "search",
                "results": results}
    if not results:
        response["message"] = "No installed nodes matched the query"
    return json.dumps(response, ensure_ascii=False)

def save_checkpoint_before_modification(session_id: str, action_description: str) -> Optional[int]:
    """在修改工作流前保存checkpoint，返回checkpoint_id"""
    try:
        current_workflow = get_workflow_data(session_id)
        if not current_workflow:
            return None
            
        checkpoint_id = save_workflow_data(
            session_id,
            current_workflow,
            workflow_data_ui=get_workflow_data_ui(session_id),
            attributes={
                "checkpoint_type": "workflow_rewrite_start",
                "description": f"Checkpoint before {action_description}",
                "action": "workflow_rewrite_checkpoint",
                "timestamp": time.time()
            }
        )
        log.info(f"Saved workflow rewrite checkpoint with ID: {checkpoint_id}")
        return checkpoint_id
    except Exception as e:
        log.error(f"Failed to save checkpoint before modification: {str(e)}")
        return None

async def node_definitions() -> Optional[Dict[str, Any]]:
    """ComfyUI's node definitions (/object_info), or None if unavailable (then only structure is checked)."""
    return await get_node_definitions()


async def apply_workflow_update(session_id: str, workflow_data: Any,
                                description: str = "Workflow structure fixed by rewrite agent") -> Dict[str, Any]:
    """Validate a rewritten API-format workflow and save it as the session's new version.

    Returns the tool-result dict: {"success", "version_id", "message", "ext"} or {"error"}.
    Invalid workflows are rejected without touching the saved workflow or the canvas.
    """
    try:
        workflow_dict = json.loads(workflow_data) if isinstance(workflow_data, str) else workflow_data
    except json.JSONDecodeError as e:
        return {"error": f"The workflow is not valid JSON ({e}). Return the complete workflow as JSON."}

    problems = validate_api_workflow(workflow_dict, await node_definitions())
    if problems:
        log.warning(f"[apply_workflow_update] rejected workflow: {problems}")
        return {"error": "The workflow was not applied because it is invalid: " + " ".join(problems[:10])}

    # 在修改前保存checkpoint
    checkpoint_id = save_checkpoint_before_modification(session_id, "workflow update")
    version_id = save_workflow_data(
        session_id,
        workflow_dict,
        attributes={"action": "workflow_rewrite", "description": description}
    )

    ext_data = [{"type": "workflow_update", "data": {"workflow_data": workflow_dict}}]
    # 如果成功保存了checkpoint，添加修改前的checkpoint信息（给用户消息）
    if checkpoint_id:
        ext_data.append({
            "type": "workflow_rewrite_checkpoint",
            "data": {"checkpoint_id": checkpoint_id, "checkpoint_type": "workflow_rewrite_start"}
        })
    if version_id:
        ext_data.append({
            "type": "workflow_rewrite_complete",
            "data": {"version_id": version_id, "checkpoint_type": "workflow_rewrite_complete"}
        })
    return {
        "success": True,
        "version_id": version_id,
        "message": f"Workflow updated successfully with version ID: {version_id}",
        "ext": ext_data,
    }


def tool_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """The default tool error function, which just returns a generic error message."""
    # return f"An error occurred while running the tool. Please try again. Error: {str(error)}"
    return json.dumps({"error": str(error)}, ensure_ascii= False)

# def update_workflow(session_id: str, workflow_data: Union[Dict[str, Any], str]) -> str:
@function_tool
async def update_workflow(workflow_data: str = "") -> str:
    """
    更新当前session的工作流数据

    Args:
        workflow_data: 工作流数据，必须是严格的json格式字符串

    Returns:
        str: 更新后的工作流数据
    """
    try:
        session_id = get_session_id()
        if not session_id:
            return json.dumps({"error": "No session_id found in context"})
        
        if not workflow_data or not isinstance(workflow_data, str) or not workflow_data.strip():
            rewrite_context = get_rewrite_context()
            log.info(f"[update_workflow] workflow_data: {workflow_data}, trigger simple rewrite, context: {rewrite_context}")
            # Blocking LLM call: run it in a thread so ComfyUI's event loop keeps serving requests
            workflow_data = await asyncio.to_thread(rewrite_workflow_simple, rewrite_context)
            
        
        return json.dumps(await apply_workflow_update(session_id, workflow_data), ensure_ascii=False)
    except Exception as e:
        log.error(f"Failed to update workflow: {str(e)}")
        return json.dumps({"error": f"Failed to update workflow: {str(e)}. Please try regenerating the workflow and then update again."})

@function_tool
def remove_node(node_id: str) -> str:
    """从工作流中移除节点"""
    try:
        session_id = get_session_id()
        if not session_id:
            return json.dumps({"error": "No session_id found in context"})
        
        # 在修改前保存checkpoint
        checkpoint_id = save_checkpoint_before_modification(session_id, f"remove node {node_id}")
        
        workflow_data = get_workflow_data(session_id)
        if not workflow_data:
            return json.dumps({"error": "No workflow data found"})
        
        if node_id not in workflow_data:
            return json.dumps({"error": f"Node {node_id} not found"})
        
        # 移除节点
        removed_node = workflow_data.pop(node_id)
        
        # 移除所有指向该节点的连接
        for other_node_id, node_data in workflow_data.items():
            inputs = node_data.get("inputs", {})
            for input_name, input_value in list(inputs.items()):
                if isinstance(input_value, list) and len(input_value) == 2:
                    if str(input_value[0]) == node_id:
                        # 移除这个连接
                        del inputs[input_name]
        
        # 保存更新
        version_id = save_workflow_data(
            session_id,
            workflow_data,
            attributes={
                "action": "remove_node",
                "description": f"Removed node {node_id}",
                "changes": {
                    "node_id": node_id,
                    "removed_node": removed_node
                }
            }
        )
        
        # 构建返回数据，包含checkpoint信息
        ext_data = [{
            "type": "workflow_update",
            "data": {
                "workflow_data": workflow_data,
                "changes": {
                    "action": "remove_node",
                    "node_id": node_id,
                    "removed_node": removed_node
                }
            }
        }]
        
        # 如果成功保存了checkpoint，添加修改前的checkpoint信息（给用户消息）
        if checkpoint_id:
            ext_data.append({
                "type": "workflow_rewrite_checkpoint",
                "data": {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_type": "workflow_rewrite_start"
                }
            })
        
        # 添加修改后的版本信息（给AI响应）
        ext_data.append({
            "type": "workflow_rewrite_complete",
            "data": {
                "version_id": version_id,
                "checkpoint_type": "workflow_rewrite_complete"
            }
        })
        
        return json.dumps({
            "success": True,
            "version_id": version_id,
            "message": f"Removed node {node_id} and cleaned up connections",
            "ext": ext_data
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to remove node: {str(e)}"})
