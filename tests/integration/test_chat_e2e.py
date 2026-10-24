"""Runs /api/chat/invoke end to end against a fake OpenAI-compatible server.

No MCP servers are configured, which is the default now that the hosted
Copilot server is gone: chat must still work with just a user-supplied LLM.
"""

import asyncio

import pytest

from backend.controller import conversation_api
from backend.service import mcp_client
from tests.fixtures.fake_llm import copilot_with_fake_llm, read_chat_lines


@pytest.fixture(autouse=True)
def no_mcp_servers(monkeypatch):
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)


def _chat(routes, headers=None, body=None, **reply):
    async def run():
        async with copilot_with_fake_llm(routes.handlers, **reply) as (client, base_url, llm_requests):
            response = await client.post("/api/chat/invoke", json=body or {
                "session_id": "e2e-session",
                "messages": [{"role": "user", "content": "hi"}],
                "ext": [{"type": "model_select", "data": ["fake-model"]}],
            }, headers={"Openai-Base-Url": base_url, **(headers or {})})
            return await read_chat_lines(response), llm_requests, base_url
    return asyncio.run(run())


def test_chat_streams_without_any_mcp_server(routes, temp_session_db):
    lines, llm_requests, _ = _chat(routes, headers={"Openai-Api-Key": "sk-user", "Authorization": "Bearer copilot-key"})

    final = lines[-1]
    assert final["finished"] is True
    assert final["text"] == "Hello there"
    assert all(line["finished"] is False for line in lines[:-1])

    (llm_request,) = llm_requests
    assert llm_request["body"]["model"] == "fake-model"
    assert llm_request["auth"] == "Bearer sk-user"
    tool_names = {tool["function"]["name"] for tool in llm_request["body"].get("tools", [])}
    # Without MCP servers the local discovery tools stand in; nothing can generate workflows
    assert {"get_current_workflow", "search_node_local", "search_models", "recall_workflow"} <= tool_names
    assert "gen_workflow" not in tool_names


def test_wrong_api_key_gives_actionable_message(routes, temp_session_db):
    lines, _, base_url = _chat(routes, headers={"Openai-Api-Key": "sk-wrong"}, status=401, error="Incorrect API key")

    final = lines[-1]
    assert final["finished"] is True
    assert "rejected the API key (HTTP 401)" in final["text"]
    assert base_url in final["text"]
    assert "sk-wrong" not in final["text"]


def test_unknown_model_gives_actionable_message(routes, temp_session_db):
    lines, _, _ = _chat(routes, status=404, error="model not found")
    assert "'fake-model' does not exist there" in lines[-1]["text"]


def test_chat_without_model_explains_configuration(routes, temp_session_db, monkeypatch):
    from backend.utils import llm_settings

    monkeypatch.setattr(llm_settings, "DEFAULT_CHAT_MODEL", None)
    lines, llm_requests, _ = _chat(routes, body={"session_id": "e2e-no-model",
                                                 "messages": [{"role": "user", "content": "hi"}]})

    assert llm_requests == []
    assert lines[-1]["finished"] is True
    assert "No model selected" in lines[-1]["text"]


@pytest.mark.parametrize("path", ["/api/chat/invoke", "/api/debug-agent"])
def test_agents_unavailable_explains_install(routes, monkeypatch, path):
    monkeypatch.setattr(conversation_api, "AGENTS_UNAVAILABLE", "Copilot's AI agents could not be loaded")

    async def run():
        async with copilot_with_fake_llm(routes.handlers) as (client, _, llm_requests):
            response = await client.post(path, json={"session_id": "s", "messages": [], "workflow_data": {}})
            return await read_chat_lines(response), llm_requests

    lines, llm_requests = asyncio.run(run())
    assert llm_requests == []
    assert lines == [{"session_id": "s", "text": "Copilot's AI agents could not be loaded", "finished": True,
                      "type": "message", "format": "markdown", "ext": None}]
