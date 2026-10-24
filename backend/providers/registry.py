"""Pick the provider for a base URL, find local servers, and cache model capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Optional

import aiohttp

from ..utils.globals import is_lmstudio_url
from .base import ModelInfo, OpenAICompatibleProvider, ProviderError, ProviderStatus, url_port
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider

PROVIDERS: dict[str, type[OpenAICompatibleProvider]] = {
    provider.kind: provider for provider in (OpenAICompatibleProvider, OllamaProvider, LMStudioProvider)
}

DISCOVERY_TIMEOUT = aiohttp.ClientTimeout(total=2)
CAPABILITY_TTL_SECONDS = 60
_LOOKUP_TIMEOUT_SECONDS = 4

_capability_cache: dict[tuple[str, str, str], tuple[float, list[ModelInfo]]] = {}


def detect_kind(base_url: Optional[str]) -> str:
    port = url_port(base_url or "")
    if port in OllamaProvider.default_ports:
        return OllamaProvider.kind
    if port in LMStudioProvider.default_ports or is_lmstudio_url(base_url or ""):
        return LMStudioProvider.kind
    return OpenAICompatibleProvider.kind


def get_provider(base_url: Optional[str], api_key: Optional[str] = None, kind: Optional[str] = None,
                 **kwargs) -> OpenAICompatibleProvider:
    provider_class = PROVIDERS.get(kind or "") or PROVIDERS[detect_kind(base_url)]
    return provider_class(base_url=base_url, api_key=api_key, **kwargs)


def provider_catalog() -> list[dict]:
    return [
        {"kind": cls.kind, "label": cls.label, "default_base_url": cls.default_base_url,
         "needs_api_key": cls.needs_api_key}
        for cls in PROVIDERS.values()
    ]


async def discover_local(host: str = "127.0.0.1") -> list[ProviderStatus]:
    """Probe the default local ports of Ollama and LM Studio; return the servers that answer."""
    candidates = [
        provider_class(base_url=f"http://{host}:{port}/v1", timeout=DISCOVERY_TIMEOUT)
        for provider_class in (OllamaProvider, LMStudioProvider)
        for port in provider_class.default_ports
    ]
    results = await asyncio.gather(*(candidate.test() for candidate in candidates))
    return [status for status in results if status.ok]


def _cache_key(provider: OpenAICompatibleProvider) -> tuple[str, str, str]:
    key_hash = hashlib.sha256((provider.api_key or "").encode()).hexdigest()[:16]
    return provider.kind, provider.base_url, key_hash


async def list_models_cached(provider: OpenAICompatibleProvider) -> list[ModelInfo]:
    key = _cache_key(provider)
    cached = _capability_cache.get(key)
    if cached and time.monotonic() - cached[0] < CAPABILITY_TTL_SECONDS:
        return cached[1]
    models = await provider.list_models()
    _capability_cache[key] = (time.monotonic(), models)
    return models


async def lookup_model(base_url: Optional[str], api_key: Optional[str], model_id: Optional[str],
                       kind: Optional[str] = None) -> Optional[ModelInfo]:
    """Capabilities of one model, or None if the provider can't be asked. Never raises."""
    if not model_id:
        return None
    provider = get_provider(base_url, api_key, kind)
    try:
        models = await asyncio.wait_for(list_models_cached(provider), _LOOKUP_TIMEOUT_SECONDS)
    except (ProviderError, asyncio.TimeoutError):
        return None
    return next((model for model in models if model.id == model_id), None)


def clear_capability_cache() -> None:
    _capability_cache.clear()
