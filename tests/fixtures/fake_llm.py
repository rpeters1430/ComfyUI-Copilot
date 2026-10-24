"""A minimal OpenAI-compatible chat-completions server for integration tests."""

import json
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

LLM_REQUESTS = web.AppKey("llm_requests", list)
LLM_REPLY = web.AppKey("llm_reply", dict)


def _chunk(delta, finish_reason=None):
    return {
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _chat_completions(request):
    body = await request.json()
    request.app[LLM_REQUESTS].append({"body": body, "auth": request.headers.get("Authorization")})
    reply = request.app[LLM_REPLY]
    if reply.get("status", 200) != 200:
        return web.json_response({"error": {"message": reply.get("error", "error"), "type": "invalid_request_error"}},
                                 status=reply["status"])

    if not body.get("stream"):
        return web.json_response({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply.get("text", "Hello there")}}],
        })

    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    words = reply.get("text", "Hello there").split(" ")
    deltas = [{"role": "assistant", "content": words[0]}] + [{"content": f" {word}"} for word in words[1:]]
    for delta in deltas:
        await response.write(f"data: {json.dumps(_chunk(delta))}\n\n".encode())
    await response.write(f"data: {json.dumps(_chunk({}, finish_reason='stop'))}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


@asynccontextmanager
async def copilot_with_fake_llm(handlers, **reply):
    """Yield (client for the Copilot routes, base URL of the fake LLM, recorded LLM requests)."""
    llm_app = web.Application()
    llm_app[LLM_REQUESTS] = []
    llm_app[LLM_REPLY] = reply
    llm_app.router.add_post("/v1/chat/completions", _chat_completions)

    copilot_app = web.Application()
    for (method, path), handler in handlers.items():
        copilot_app.router.add_route(method, path, handler)

    async with TestServer(llm_app) as llm_server, TestClient(TestServer(copilot_app)) as client:
        yield client, str(llm_server.make_url("/v1")), llm_app[LLM_REQUESTS]


async def read_chat_lines(response):
    return [json.loads(line) for line in (await response.text()).splitlines() if line.strip()]
