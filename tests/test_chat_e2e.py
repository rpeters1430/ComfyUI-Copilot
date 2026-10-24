"""Runs /api/chat/invoke end to end against a fake OpenAI-compatible server.

No MCP servers are configured, which is the default now that the hosted
Copilot server is gone: chat must still work with just a user-supplied LLM.
"""

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from backend.service import mcp_client

LLM_REQUESTS = web.AppKey("llm_requests", list)


def _chunk(delta, finish_reason=None):
    return {
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _fake_chat_completions(request):
    body = await request.json()
    request.app[LLM_REQUESTS].append({"body": body, "auth": request.headers.get("Authorization")})
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for chunk in (_chunk({"role": "assistant", "content": "Hello "}), _chunk({"content": "there"}),
                  _chunk({}, finish_reason="stop")):
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


async def _run_chat(routes, temp_session_db, headers, body):
    llm_app = web.Application()
    llm_app[LLM_REQUESTS] = []
    llm_app.router.add_post("/v1/chat/completions", _fake_chat_completions)

    copilot_app = web.Application()
    copilot_app.router.add_post("/api/chat/invoke", routes.handlers[("POST", "/api/chat/invoke")])

    async with TestServer(llm_app) as llm_server, TestClient(TestServer(copilot_app)) as client:
        base_url = str(llm_server.make_url("/v1"))
        response = await client.post("/api/chat/invoke", json=body,
                                     headers={"Openai-Base-Url": base_url, **headers})
        lines = [json.loads(line) for line in (await response.text()).splitlines() if line.strip()]
    return lines, llm_app[LLM_REQUESTS]


def test_chat_streams_without_any_mcp_server(routes, temp_session_db, monkeypatch):
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)

    lines, llm_requests = asyncio.run(_run_chat(
        routes, temp_session_db,
        headers={"Openai-Api-Key": "sk-user", "Authorization": "Bearer copilot-key"},
        body={
            "session_id": "e2e-session",
            "messages": [{"role": "user", "content": "hi"}],
            "ext": [{"type": "model_select", "data": ["fake-model"]}],
        },
    ))

    final = lines[-1]
    assert final["finished"] is True
    assert final["text"] == "Hello there"
    assert all(line["finished"] is False for line in lines[:-1])

    (llm_request,) = llm_requests
    assert llm_request["body"]["model"] == "fake-model"
    assert llm_request["auth"] == "Bearer sk-user"
    tool_names = {tool["function"]["name"] for tool in llm_request["body"].get("tools", [])}
    assert {"get_current_workflow", "search_node_local"} <= tool_names
    assert not tool_names & {"recall_workflow", "gen_workflow"}


def test_chat_without_model_explains_configuration(routes, temp_session_db, monkeypatch):
    from backend import agent_factory

    monkeypatch.setattr(agent_factory, "DEFAULT_CHAT_MODEL", None)
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)

    lines, llm_requests = asyncio.run(_run_chat(
        routes, temp_session_db, headers={},
        body={"session_id": "e2e-no-model", "messages": [{"role": "user", "content": "hi"}]},
    ))

    assert llm_requests == []
    assert lines[-1]["finished"] is True
    assert "No model selected" in lines[-1]["text"]
