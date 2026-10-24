"""Provider interface: discover models and what they can do.

Every provider is reached for inference through its OpenAI-compatible
endpoint. The provider classes only add discovery (which models exist,
whether they support tools, images, structured output, how much context)
on top of that, using each server's native API where it has one.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=8)


@dataclass
class ModelCapabilities:
    """What a model supports. None means the provider doesn't say."""

    tools: Optional[bool] = None
    vision: Optional[bool] = None
    structured_output: Optional[bool] = None
    context_length: Optional[int] = None


@dataclass
class ModelInfo:
    id: str
    provider: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    # "llm", "vlm" or "embedding"; embeddings are never offered as chat models
    type: str = "llm"
    display_name: Optional[str] = None
    loaded: Optional[bool] = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_chat_model(self) -> bool:
        return self.type != "embedding"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderStatus:
    ok: bool
    kind: str
    base_url: str
    message: str
    version: Optional[str] = None
    models: list[ModelInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["models"] = [model.to_dict() for model in self.models]
        return data


class ProviderError(Exception):
    """The provider is unreachable or answered with an error."""

    def __init__(self, message: str, status: Optional[int] = None, unreachable: bool = False):
        super().__init__(message)
        self.status = status
        self.unreachable = unreachable


def server_root(base_url: str) -> str:
    """Base URL without a trailing /v1 (where native APIs such as Ollama's /api live)."""
    url = (base_url or "").strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


class OpenAICompatibleProvider:
    """Any server exposing GET /v1/models (OpenAI, OpenRouter, vLLM, llama.cpp, ...)."""

    kind = "openai"
    label = "OpenAI-compatible"
    default_base_url = "https://api.openai.com/v1"
    needs_api_key = True

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: aiohttp.ClientTimeout = DEFAULT_TIMEOUT):
        self.base_url = (base_url or self.default_base_url).strip().rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout

    @property
    def openai_base_url(self) -> str:
        """URL to hand to the OpenAI client for inference."""
        return self.base_url

    @property
    def root(self) -> str:
        return server_root(self.base_url)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _request(self, method: str, url: str, json_body: Any = None) -> Any:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.request(method, url, json=json_body, headers=self._headers()) as response:
                    if response.status == 401:
                        raise ProviderError("The server rejected the API key (HTTP 401).", status=401)
                    if response.status >= 400:
                        text = (await response.text())[:300]
                        raise ProviderError(f"HTTP {response.status} from {url}: {text}", status=response.status)
                    return await response.json(content_type=None)
        except asyncio.TimeoutError as e:
            raise ProviderError(f"Timed out connecting to {url}.", unreachable=True) from e
        except aiohttp.ClientError as e:
            raise ProviderError(f"Could not connect to {url} ({type(e).__name__}).", unreachable=True) from e

    async def _openai_models(self) -> list[ModelInfo]:
        payload = await self._request("GET", f"{self.openai_base_url}/models")
        models = []
        for item in (payload or {}).get("data", []):
            model_id = item.get("id")
            if model_id:
                models.append(ModelInfo(id=model_id, provider=self.kind, type=_guess_type(model_id)))
        return models

    async def list_models(self) -> list[ModelInfo]:
        return await self._openai_models()

    async def version(self) -> Optional[str]:
        return None

    def warnings(self, models: list[ModelInfo]) -> list[str]:
        return []

    async def test(self) -> ProviderStatus:
        """Check the connection and report models, version and setup warnings."""
        try:
            models = await self.list_models()
            version = await self.version()
        except ProviderError as e:
            return ProviderStatus(ok=False, kind=self.kind, base_url=self.base_url, message=str(e))
        chat_models = [model for model in models if model.is_chat_model]
        warnings = self.warnings(models)
        if not chat_models:
            warnings.append("The server is reachable but has no chat models available.")
        noun = "model" if len(chat_models) == 1 else "models"
        return ProviderStatus(ok=True, kind=self.kind, base_url=self.base_url, version=version,
                              message=f"Connected to {self.label}: {len(chat_models)} chat {noun} available.",
                              models=models, warnings=warnings)


def _guess_type(model_id: str) -> str:
    return "embedding" if "embed" in model_id.lower() else "llm"


def url_port(base_url: str) -> Optional[int]:
    try:
        return urlparse(base_url).port
    except ValueError:
        return None
