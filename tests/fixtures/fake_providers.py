"""Fake Ollama and LM Studio servers that answer with their native API shapes."""

from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

OLLAMA_TAGS = {"models": [
    {"name": "llama3.2:latest", "model": "llama3.2:latest", "size": 2019393189,
     "details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"}},
    {"name": "qwen2.5vl:7b", "model": "qwen2.5vl:7b", "size": 6000000000,
     "details": {"family": "qwen25vl", "parameter_size": "8.3B", "quantization_level": "Q4_K_M"}},
    {"name": "gemma3:4b", "model": "gemma3:4b", "size": 3300000000,
     "details": {"family": "gemma3", "parameter_size": "4.3B", "quantization_level": "Q4_K_M"}},
    {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "size": 274302450,
     "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"}},
]}

OLLAMA_SHOW = {
    "llama3.2:latest": {"capabilities": ["completion", "tools"],
                        "model_info": {"general.architecture": "llama", "llama.context_length": 131072}},
    "qwen2.5vl:7b": {"capabilities": ["completion", "vision", "tools"],
                     "model_info": {"general.architecture": "qwen25vl", "qwen25vl.context_length": 128000}},
    "gemma3:4b": {"capabilities": ["completion", "vision"],
                  "model_info": {"general.architecture": "gemma3", "gemma3.context_length": 131072}},
    "nomic-embed-text:latest": {"capabilities": ["embedding"],
                                "model_info": {"general.architecture": "nomic-bert", "nomic-bert.context_length": 2048}},
}


@asynccontextmanager
async def fake_ollama(version="0.12.3", loaded=None):
    """loaded: {model name: context_length} for /api/ps."""
    app = web.Application()

    async def tags(request):
        return web.json_response(OLLAMA_TAGS)

    async def show(request):
        name = (await request.json())["model"]
        if name not in OLLAMA_SHOW:
            raise web.HTTPNotFound()
        return web.json_response(OLLAMA_SHOW[name])

    async def ps(request):
        return web.json_response({"models": [{"name": name, "context_length": ctx}
                                             for name, ctx in (loaded or {}).items()]})

    async def get_version(request):
        return web.json_response({"version": version})

    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/show", show)
    app.router.add_get("/api/ps", ps)
    app.router.add_get("/api/version", get_version)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), server.port


@asynccontextmanager
async def fake_lmstudio(api="v1"):
    """api: "v1" (0.3.x native API), "v0" (older native API) or "openai" (/v1/models only)."""
    app = web.Application()

    async def v1(request):
        return web.json_response({"models": [
            {"type": "llm", "key": "qwen2.5-coder-7b", "display_name": "Qwen2.5 Coder 7B",
             "max_context_length": 32768, "loaded_instances": [{"id": "x"}],
             "capabilities": {"vision": False, "trained_for_tool_use": True}},
            {"type": "embedding", "key": "nomic-embed", "max_context_length": 2048, "loaded_instances": []},
        ]})

    async def v0(request):
        return web.json_response({"data": [
            {"id": "llava-1.6", "type": "vlm", "state": "loaded", "max_context_length": 4096},
            {"id": "nomic-embed", "type": "embeddings", "state": "not-loaded", "max_context_length": 2048},
        ]})

    async def openai_models(request):
        return web.json_response({"data": [{"id": "some-model"}, {"id": "text-embedding-x"}]})

    if api == "v1":
        app.router.add_get("/api/v1/models", v1)
    if api in ("v1", "v0"):
        app.router.add_get("/api/v0/models", v0)
    app.router.add_get("/v1/models", openai_models)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), server.port
