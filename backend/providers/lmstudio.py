"""LM Studio: native REST API for capabilities, falling back to older APIs.

LM Studio 0.3.x exposes /api/v1/models (with vision/tool-use flags and
max context), earlier versions /api/v0/models (type and context only),
and every version the OpenAI-compatible /v1/models (ids only).
"""

from __future__ import annotations

from .base import ModelCapabilities, ModelInfo, OpenAICompatibleProvider, ProviderError

_TYPES = {"llm": "llm", "vlm": "vlm", "embedding": "embedding", "embeddings": "embedding"}


class LMStudioProvider(OpenAICompatibleProvider):
    kind = "lmstudio"
    label = "LM Studio"
    default_base_url = "http://localhost:1234/v1"
    needs_api_key = False
    default_ports = (1234, 1235)

    @property
    def openai_base_url(self) -> str:
        return f"{self.root}/v1"

    async def _v1_models(self) -> list[ModelInfo]:
        payload = await self._request("GET", f"{self.root}/api/v1/models")
        items = (payload or {}).get("models")
        if not isinstance(items, list):
            raise ProviderError("Unexpected /api/v1/models response")
        models = []
        for item in items:
            caps = item.get("capabilities") or {}
            model_type = _TYPES.get(item.get("type"), "llm")
            models.append(ModelInfo(
                id=item.get("key"),
                provider=self.kind,
                type=model_type,
                display_name=item.get("display_name"),
                loaded=bool(item.get("loaded_instances")),
                capabilities=ModelCapabilities(
                    # LM Studio can still prompt untrained models into tool calls, but unreliably
                    tools=caps.get("trained_for_tool_use"),
                    vision=caps.get("vision", model_type == "vlm"),
                    structured_output=True,
                    context_length=item.get("max_context_length"),
                ),
                details={
                    "architecture": item.get("architecture"),
                    "parameters": item.get("params_string"),
                    "quantization": (item.get("quantization") or {}).get("name"),
                },
            ))
        return [model for model in models if model.id]

    async def _v0_models(self) -> list[ModelInfo]:
        payload = await self._request("GET", f"{self.root}/api/v0/models")
        models = []
        for item in (payload or {}).get("data", []):
            model_type = _TYPES.get(item.get("type"), "llm")
            models.append(ModelInfo(
                id=item.get("id"),
                provider=self.kind,
                type=model_type,
                loaded=item.get("state") == "loaded",
                capabilities=ModelCapabilities(
                    vision=model_type == "vlm",
                    structured_output=True,
                    context_length=item.get("max_context_length"),
                ),
                details={"architecture": item.get("arch"), "quantization": item.get("quantization")},
            ))
        return [model for model in models if model.id]

    async def list_models(self) -> list[ModelInfo]:
        for source in (self._v1_models, self._v0_models):
            try:
                return await source()
            except ProviderError as e:
                # Older LM Studio lacks the newer API (404); a down server or bad key won't recover
                if e.unreachable or e.status in (401, 403):
                    raise
        return await self._openai_models()
