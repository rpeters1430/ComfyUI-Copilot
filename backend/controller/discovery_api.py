"""Routes for what this ComfyUI has: installed nodes, model files and the local workflow library."""

import asyncio

from aiohttp import web
import server

from ..discovery.library import recall_local_workflows
from ..discovery.models import list_models, search_models
from ..discovery.nodes import find_class, get_node_definitions, search_nodes, summarize_node
from ..discovery import workflows


def _limit(request, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(request.query.get("limit", default)), maximum))
    except ValueError:
        return default


@server.PromptServer.instance.routes.get("/api/discovery/nodes")
async def discover_nodes(request):
    """?q=keywords searches installed nodes; ?class=Name returns that node's compact spec."""
    definitions = await get_node_definitions()
    if not definitions:
        return web.json_response({"success": False, "message": "ComfyUI did not return node definitions"},
                                 status=503)
    wanted = request.query.get("class", "").strip()
    if wanted:
        class_name = find_class(definitions, wanted)
        if class_name is None:
            return web.json_response({"success": False, "message": f"Node '{wanted}' is not installed",
                                      "data": search_nodes(definitions, wanted, 5)}, status=404)
        return web.json_response({"success": True, "data": summarize_node(class_name, definitions[class_name])})
    return web.json_response({"success": True, "total": len(definitions),
                              "data": search_nodes(definitions, request.query.get("q", ""), _limit(request, 20, 100))})


@server.PromptServer.instance.routes.get("/api/discovery/models")
async def discover_models(request):
    """?q= searches installed model files, optionally within ?folder=; no query lists counts per folder."""
    folder = request.query.get("folder", "").strip() or None
    inventory = await asyncio.to_thread(list_models, folder)
    query = request.query.get("q", "")
    if not query and not folder:
        return web.json_response({"success": True, "data": {name: len(files) for name, files in inventory.items()}})
    results = search_models(query, inventory=inventory, limit=_limit(request, 50, 500))
    return web.json_response({"success": True, "data": results})


@server.PromptServer.instance.routes.get("/api/discovery/workflows")
async def discover_workflows(request):
    """The local workflow library, searched with ?q=, with missing nodes and models per workflow."""
    result = await recall_local_workflows(request.query.get("q", ""), _limit(request, 20, 200))
    return web.json_response({"success": True, "data": result["workflows"]})


@server.PromptServer.instance.routes.get("/api/discovery/workflows/{workflow_id}")
async def get_library_workflow(request):
    """One library workflow's JSON, by the id from a search result (never by path)."""
    entry = await asyncio.to_thread(workflows.workflow_index.get, request.match_info["workflow_id"])
    if entry is None:
        return web.json_response({"success": False, "message": "Workflow not found"}, status=404)
    try:
        workflow = await asyncio.to_thread(workflows.load_workflow, entry)
    except (OSError, ValueError) as e:
        return web.json_response({"success": False, "message": f"Could not read the workflow: {e}"}, status=500)
    return web.json_response({"success": True, "data": {**entry.public(), "workflow": workflow}})


@server.PromptServer.instance.routes.get("/api/discovery/workflows/{workflow_id}/thumbnail")
async def get_library_workflow_thumbnail(request):
    entry = await asyncio.to_thread(workflows.workflow_index.get, request.match_info["workflow_id"])
    if entry is None or not entry.thumbnail:
        return web.Response(status=404)
    return web.FileResponse(entry.thumbnail)
