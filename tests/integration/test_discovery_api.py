"""Discovery routes, the local recall_workflow tool in chat, and rewrites checked against installed nodes."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from backend.service import mcp_client
from backend.service.workflow_rewrite_tools import apply_workflow_update
from tests.fixtures.comfy_env import PONY_UI, TXT2IMG_API, install, write_workflow
from tests.fixtures.fake_llm import copilot_with_fake_llm, read_chat_lines, tool_call


@pytest.fixture
def comfy(monkeypatch, tmp_path, temp_session_db, temp_workflow_db):
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)
    return install(monkeypatch, tmp_path)


def _get(routes, path):
    async def run():
        app = web.Application()
        for (method, route), handler in routes.handlers.items():
            app.router.add_route(method, route, handler)
        async with TestClient(TestServer(app)) as client:
            response = await client.get(path)
            body = await response.read()
            try:
                return response.status, json.loads(body)
            except ValueError:
                return response.status, body
    return asyncio.run(run())


def test_nodes_route_searches_and_describes(routes, comfy):
    status, body = _get(routes, "/api/discovery/nodes?q=upscale")
    assert status == 200 and body["data"][0]["class_name"] == "UltimateSDUpscale"
    status, body = _get(routes, "/api/discovery/nodes?class=ksampler")
    assert status == 200 and body["data"]["class_name"] == "KSampler"
    status, body = _get(routes, "/api/discovery/nodes?class=FaceDetailer")
    assert status == 404 and "not installed" in body["message"]


def test_models_route_counts_and_searches(routes, comfy):
    status, body = _get(routes, "/api/discovery/models")
    assert status == 200 and body["data"]["checkpoints"] == 2
    _, body = _get(routes, "/api/discovery/models?q=sdxl&folder=vae")
    assert body["data"] == [{"folder": "vae", "name": "sdxl_vae.safetensors"}]


def test_workflow_routes_list_and_load_by_id_only(routes, comfy):
    user_dir, extra = comfy
    write_workflow(user_dir, "SDXL txt2img.json", TXT2IMG_API)
    write_workflow(extra, "pony portrait.json", PONY_UI)
    (extra / "pony portrait.png").write_bytes(b"\x89PNG")

    _, body = _get(routes, "/api/discovery/workflows?q=pony")
    (pony,) = body["data"]
    assert pony["missing_nodes"] == ["FaceDetailer"]
    assert pony["missing_models"] == ["missing_style_lora.safetensors"]
    assert "path" not in pony

    status, body = _get(routes, f"/api/discovery/workflows/{pony['id']}")
    assert status == 200 and body["data"]["workflow"] == PONY_UI
    status, thumbnail = _get(routes, f"/api/discovery/workflows/{pony['id']}/thumbnail")
    assert status == 200 and thumbnail == b"\x89PNG"
    assert _get(routes, "/api/discovery/workflows/..%2F..%2Fsecrets")[0] == 404


def test_chat_recall_workflow_shows_local_workflows_as_cards(routes, comfy):
    user_dir, extra = comfy
    write_workflow(user_dir, "SDXL txt2img.json", TXT2IMG_API)
    write_workflow(extra, "pony portrait.json", PONY_UI)

    async def run():
        async with copilot_with_fake_llm(routes.handlers, texts=[
            tool_call("recall_workflow", query="pony portrait"),
            "Here is a Pony portrait workflow from your library.",
        ]) as (client, base_url, llm_requests):
            response = await client.post("/api/chat/invoke", json={
                "session_id": "recall-session",
                "messages": [{"role": "user", "content": "find me a pony portrait workflow"}],
                "ext": [{"type": "model_select", "data": ["fake-model"]}],
            }, headers={"Openai-Base-Url": base_url})
            return await read_chat_lines(response), llm_requests

    lines, llm_requests = asyncio.run(run())
    final = lines[-1]
    assert final["finished"] is True
    (cards,) = [item["data"] for item in final["ext"] if item["type"] == "workflow"]
    assert [card["name"] for card in cards] == ["pony portrait"]
    assert cards[0]["source"] == "Local library"
    assert "Missing nodes: FaceDetailer." in cards[0]["description"]
    # The model saw the tool result, including what the workflow is missing
    tool_messages = [m for m in llm_requests[1]["body"]["messages"] if m["role"] == "tool"]
    assert "missing_style_lora.safetensors" in tool_messages[0]["content"]


def test_rewrite_with_wrong_input_type_leaves_canvas_unchanged(comfy, temp_workflow_db):
    temp_workflow_db.save_workflow_data("type-session", TXT2IMG_API)
    broken = json.loads(json.dumps(TXT2IMG_API))
    broken["3"]["inputs"]["model"] = ["4", 1]  # CLIP into MODEL
    broken["4"]["inputs"]["ckpt_name"] = "not_installed.safetensors"

    result = asyncio.run(apply_workflow_update("type-session", broken))
    assert "expects MODEL but is linked to a CLIP output" in result["error"]
    assert "'not_installed.safetensors', which is not an available option" in result["error"]
    assert temp_workflow_db.get_workflow_data("type-session") == TXT2IMG_API

    assert asyncio.run(apply_workflow_update("type-session", TXT2IMG_API))["success"] is True
