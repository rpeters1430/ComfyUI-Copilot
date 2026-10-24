import asyncio
import contextlib
import json
import types

import pytest

from backend.service import mcp_client
from backend.service.mcp_client import (
    build_router_instructions,
    connect_mcp_servers,
    merge_workflow_results,
    parse_tool_output,
)


def _result(data=None, ext=None):
    return {"answer": None, "data": data, "ext": ext, "content_dict": None}


def test_merge_puts_generated_first_and_dedupes():
    tool_results = {
        "recall_workflow": _result(data=[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]),
        "gen_workflow": _result(data=[{"id": 9, "name": "gen1"}, {"id": 2, "name": "dup"}]),
    }
    ext, finished = merge_workflow_results(tool_results, disable_gen=False)
    assert finished is True
    assert [w["id"] for w in ext[0]["data"]] == [9, 2, 1]


def test_merge_waits_for_gen_when_only_recall_ran():
    ext, finished = merge_workflow_results({"recall_workflow": _result(data=[{"id": 1}])}, disable_gen=False)
    assert (ext, finished) == (None, False)


def test_merge_returns_recall_when_generation_disabled():
    ext, finished = merge_workflow_results({"recall_workflow": _result(data=[{"id": 1}])}, disable_gen=True)
    assert finished is True
    assert ext == [{"type": "workflow", "data": [{"id": 1}]}]


def test_merge_without_workflow_tools_uses_first_tool_ext():
    ext, finished = merge_workflow_results({
        "get_current_workflow": _result(),
        "search_node_local": _result(ext=[{"type": "node", "data": []}]),
    })
    assert finished is True
    assert ext == [{"type": "node", "data": []}]


def test_parse_tool_output_extracts_ext_and_nested_text():
    raw = json.dumps({
        "ext": [{"type": "workflow_update", "data": {}}],
        "text": json.dumps({"answer": "ok", "data": [1], "ext": [{"type": "x"}]}),
    })
    parsed = parse_tool_output(raw)
    assert parsed["raw_ext"] == [{"type": "workflow_update", "data": {}}]
    assert parsed["answer"] == "ok"
    assert parsed["data"] == [1]
    assert parsed["ext"] == [{"type": "x"}]


def test_parse_tool_output_handles_plain_text():
    parsed = parse_tool_output("not json")
    assert parsed["answer"] == "not json"
    assert parsed["raw_ext"] is None


def test_instructions_without_remote_tools_do_not_mention_them():
    text = build_router_instructions(set())
    assert "recall_workflow" not in text
    assert "gen_workflow" not in text
    assert "bing_search" not in text
    assert "search_node_local" in text


def test_instructions_with_remote_tools(monkeypatch):
    monkeypatch.setattr(mcp_client, "DISABLE_WORKFLOW_GEN", False)
    text = build_router_instructions({"recall_workflow", "gen_workflow", "bing_search"})
    assert "BOTH recall_workflow tool AND gen_workflow" in text
    assert "`bing_search`" in text


class _FakeServer:
    def __init__(self, name, fail=False, hang=False):
        self.name = name
        self.fail = fail
        self.hang = hang
        self.cleaned = False

    async def connect(self):
        if self.hang:
            await asyncio.sleep(10)
        if self.fail:
            raise ConnectionError("503 Service Suspended")

    async def list_tools(self):
        return [types.SimpleNamespace(name=f"{self.name}_tool")]

    async def cleanup(self):
        self.cleaned = True


def test_connect_skips_unreachable_servers():
    good, bad, slow = _FakeServer("good"), _FakeServer("bad", fail=True), _FakeServer("slow", hang=True)

    async def main():
        async with contextlib.AsyncExitStack() as stack:
            connected, tools = await connect_mcp_servers(stack, [bad, good, slow], timeout=0.1)
            assert connected == [good]
            assert tools == {"good_tool"}
        assert good.cleaned

    asyncio.run(main())
    assert bad.cleaned and slow.cleaned


def test_no_mcp_servers_configured_by_default(monkeypatch):
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)
    assert mcp_client.build_mcp_servers("session") == []


def test_search_server_never_gets_copilot_key(monkeypatch):
    from backend.utils.request_context import set_request_context

    set_request_context("s", None, {"copilot_api_key": "secret-copilot-key"})
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", "https://search.test/sse")
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_API_KEY", None)
    (server,) = mcp_client.build_mcp_servers("session")
    assert "secret-copilot-key" not in json.dumps(server.params.get("headers", {}))


@pytest.mark.parametrize("error,expected", [
    (ConnectionError("Connection reset by peer"), True),
    (ValueError("model not found"), False),
])
def test_is_retryable_stream_error(error, expected):
    assert mcp_client.is_retryable_stream_error(error) is expected
