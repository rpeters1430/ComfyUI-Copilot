"""Routes for configuring LLM providers: list kinds, find local servers, test a connection."""

from aiohttp import web
import server

from ..providers import discover_local, get_provider, provider_catalog
from ..providers.registry import clear_capability_cache
from ..utils.logger import log
from ..utils.request_utils import read_json


@server.PromptServer.instance.routes.get("/api/providers")
async def list_providers(request):
    return web.json_response({"success": True, "data": provider_catalog()})


@server.PromptServer.instance.routes.get("/api/providers/discover")
async def discover_providers(request):
    """Ollama and LM Studio servers answering on their default local ports."""
    statuses = await discover_local()
    log.info(f"Discovered local providers: {[status.base_url for status in statuses]}")
    return web.json_response({"success": True, "data": [status.to_dict() for status in statuses]})


@server.PromptServer.instance.routes.post("/api/providers/test")
async def test_provider(request):
    """Connect to a provider and report its models with capabilities, plus setup warnings."""
    body = await read_json(request)
    if body is None or not str(body.get("base_url") or "").strip():
        return web.json_response({"success": False, "message": "Missing 'base_url'"}, status=400)

    # The key comes in a header, like the other LLM routes, so it isn't logged with the body
    provider = get_provider(body["base_url"], request.headers.get("Openai-Api-Key"), body.get("kind"))
    # A test means "look again": drop cached capabilities for every provider
    clear_capability_cache()
    status = await provider.test()
    return web.json_response({"success": status.ok, "data": status.to_dict(), "message": status.message})
