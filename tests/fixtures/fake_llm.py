"""A minimal OpenAI-compatible server for integration tests.

It can also impersonate LM Studio's and Ollama's native model APIs so
capability detection can be tested without either installed.
"""

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


def tool_call(name, **arguments):
    """A queued reply (in "texts") that makes the model call a tool instead of answering."""
    return {"tool_call": {"name": name, "arguments": arguments}}


def _next_text(reply):
    # "texts" is a queue of replies for multi-step flows; "text" is the reply when the queue is empty
    texts = reply.get("texts")
    if texts:
        return texts.pop(0)
    return reply.get("text", "Hello there")


async def _chat_completions(request):
    body = await request.json()
    request.app[LLM_REQUESTS].append({"body": body, "auth": request.headers.get("Authorization")})
    reply = request.app[LLM_REPLY]
    if reply.get("status", 200) != 200:
        return web.json_response({"error": {"message": reply.get("error", "error"), "type": "invalid_request_error"}},
                                 status=reply["status"])

    text = _next_text(reply)
    if isinstance(text, dict) and "tool_call" in text:
        return await _stream_tool_call(request, text["tool_call"])
    if not body.get("stream"):
        return web.json_response({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        })

    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    words = text.split(" ")
    deltas = [{"role": "assistant", "content": words[0]}] + [{"content": f" {word}"} for word in words[1:]]
    for delta in deltas:
        await response.write(f"data: {json.dumps(_chunk(delta))}\n\n".encode())
    await response.write(f"data: {json.dumps(_chunk({}, finish_reason='stop'))}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


async def _stream_tool_call(request, call):
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    delta = {"role": "assistant", "tool_calls": [{
        "index": 0, "id": f"call_{call['name']}", "type": "function",
        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
    }]}
    await response.write(f"data: {json.dumps(_chunk(delta))}\n\n".encode())
    await response.write(f"data: {json.dumps(_chunk({}, finish_reason='tool_calls'))}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


async def _openai_models(request):
    models = request.app[LLM_REPLY].get("models", [])
    return web.json_response({"object": "list", "data": [{"id": m["key"], "object": "model"} for m in models]})


async def _lmstudio_v1_models(request):
    models = request.app[LLM_REPLY].get("models")
    if models is None:
        raise web.HTTPNotFound()
    return web.json_response({"models": models})


def lmstudio_model(key, tools=False, vision=False, context=32768, model_type="llm"):
    return {"type": model_type, "key": key, "display_name": key, "max_context_length": context,
            "loaded_instances": [], "capabilities": {"vision": vision, "trained_for_tool_use": tools}}


@asynccontextmanager
async def fake_llm_server(**reply):
    """Yield (base_url ending in /v1, recorded chat requests)."""
    app = web.Application()
    app[LLM_REQUESTS] = []
    app[LLM_REPLY] = reply
    app.router.add_post("/v1/chat/completions", _chat_completions)
    app.router.add_get("/v1/models", _openai_models)
    app.router.add_get("/api/v1/models", _lmstudio_v1_models)
    async with TestServer(app) as server:
        yield str(server.make_url("/v1")), app[LLM_REQUESTS]


@asynccontextmanager
async def copilot_with_fake_llm(handlers, **reply):
    """Yield (client for the Copilot routes, base URL of the fake LLM, recorded LLM requests)."""
    copilot_app = web.Application()
    for (method, path), handler in handlers.items():
        copilot_app.router.add_route(method, path, handler)

    async with fake_llm_server(**reply) as (base_url, llm_requests), TestClient(TestServer(copilot_app)) as client:
        yield client, base_url, llm_requests


async def read_chat_lines(response):
    return [json.loads(line) for line in (await response.text()).splitlines() if line.strip()]
