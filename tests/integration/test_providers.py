import asyncio
import socket

import pytest

from backend.providers import detect_kind, get_provider, registry
from backend.providers.lmstudio import LMStudioProvider
from backend.providers.ollama import OllamaProvider
from tests.fixtures.fake_providers import fake_lmstudio, fake_ollama


def run(coro):
    return asyncio.run(coro)


def _by_id(models):
    return {model.id: model for model in models}


def test_ollama_capabilities_from_show():
    async def main():
        async with fake_ollama() as (root, _):
            return await OllamaProvider(f"{root}/v1").list_models()

    models = _by_id(run(main()))
    assert models["llama3.2:latest"].capabilities.tools is True
    assert models["llama3.2:latest"].capabilities.vision is False
    assert models["llama3.2:latest"].capabilities.context_length == 131072
    assert models["qwen2.5vl:7b"].type == "vlm"
    # gemma3 sees images but can't call tools: the JSON fallback case
    assert models["gemma3:4b"].capabilities.tools is False
    assert models["gemma3:4b"].capabilities.vision is True
    assert models["nomic-embed-text:latest"].type == "embedding"
    assert models["llama3.2:latest"].capabilities.structured_output is True


def test_ollama_accepts_base_url_with_or_without_v1():
    async def main():
        async with fake_ollama() as (root, _):
            return [len(await OllamaProvider(url).list_models()) for url in (root, f"{root}/v1", f"{root}/v1/")]

    assert run(main()) == [4, 4, 4]


def test_ollama_warns_about_small_loaded_context():
    async def main():
        async with fake_ollama(loaded={"llama3.2:latest": 4096}) as (root, _):
            return await OllamaProvider(root).test()

    status = run(main())
    assert status.ok
    assert "OLLAMA_CONTEXT_LENGTH" in status.warnings[0]
    assert "3 chat models" in status.message


def test_old_ollama_without_capabilities_reports_unknown():
    async def main():
        async with fake_ollama(version="0.3.0") as (root, _):
            provider = OllamaProvider(root)

            async def no_capabilities(name):
                return {"model_info": {}}

            provider._show = no_capabilities
            return await provider.list_models()

    model = _by_id(run(main()))["llama3.2:latest"]
    assert model.capabilities.tools is None
    assert model.capabilities.structured_output is None


@pytest.mark.parametrize("api,expected", [
    ("v1", {"qwen2.5-coder-7b": ("llm", True, False, True)}),
    ("v0", {"llava-1.6": ("vlm", None, True, True)}),
    ("openai", {"some-model": ("llm", None, None, None)}),
])
def test_lmstudio_falls_back_through_apis(api, expected):
    async def main():
        async with fake_lmstudio(api) as (root, _):
            return await LMStudioProvider(f"{root}/v1").list_models()

    models = {model.id: model for model in run(main()) if model.is_chat_model}
    assert {model_id: (m.type, m.capabilities.tools, m.capabilities.vision, m.loaded)
            for model_id, m in models.items()} == expected


def _closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("provider_class", [OllamaProvider, LMStudioProvider])
def test_unreachable_provider_fails_gracefully(provider_class):
    status = run(provider_class(f"http://127.0.0.1:{_closed_port()}/v1").test())
    assert status.ok is False
    assert "Could not connect" in status.message


@pytest.mark.parametrize("url,kind", [
    ("http://localhost:11434/v1", "ollama"),
    ("http://127.0.0.1:11434", "ollama"),
    ("http://localhost:1234/v1", "lmstudio"),
    ("http://192.168.1.20:1235/v1", "lmstudio"),
    ("https://api.openai.com/v1", "openai"),
    ("https://openrouter.ai/api/v1", "openai"),
])
def test_detect_kind(url, kind):
    assert detect_kind(url) == kind


def test_explicit_kind_wins_over_url():
    assert get_provider("http://my-box:9000/v1", kind="ollama").kind == "ollama"


def test_discover_local_finds_running_servers(monkeypatch):
    async def main():
        async with fake_ollama() as (_, ollama_port), fake_lmstudio() as (_, lmstudio_port):
            monkeypatch.setattr(OllamaProvider, "default_ports", (ollama_port,))
            monkeypatch.setattr(LMStudioProvider, "default_ports", (_closed_port(), lmstudio_port))
            return await registry.discover_local()

    statuses = run(main())
    assert sorted(status.kind for status in statuses) == ["lmstudio", "ollama"]


def test_lookup_model_never_raises_and_caches():
    registry.clear_capability_cache()

    async def main():
        missing = await registry.lookup_model(f"http://127.0.0.1:{_closed_port()}/v1", None, "x")
        async with fake_lmstudio() as (root, _):
            found = await registry.lookup_model(f"{root}/v1", None, "qwen2.5-coder-7b", "lmstudio")
        # The server is gone now; the cached capabilities still answer
        cached = await registry.lookup_model(f"{root}/v1", None, "qwen2.5-coder-7b", "lmstudio")
        return missing, found, cached

    missing, found, cached = run(main())
    assert missing is None
    assert found.capabilities.tools is True
    assert cached is not None
