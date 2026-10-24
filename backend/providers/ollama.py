"""Ollama: models from /api/tags, capabilities and context length from /api/show."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .base import ModelCapabilities, ModelInfo, OpenAICompatibleProvider, ProviderError

# Workflow JSON plus instructions easily exceeds small context windows
MIN_WORKFLOW_CONTEXT = 16384
_SHOW_CONCURRENCY = 4


def _version_tuple(version: Optional[str]) -> tuple[int, ...]:
    parts = []
    for piece in (version or "").split("-")[0].split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


class OllamaProvider(OpenAICompatibleProvider):
    kind = "ollama"
    label = "Ollama"
    default_base_url = "http://localhost:11434/v1"
    needs_api_key = False
    default_ports = (11434,)

    @property
    def openai_base_url(self) -> str:
        return f"{self.root}/v1"

    async def version(self) -> Optional[str]:
        try:
            return (await self._request("GET", f"{self.root}/api/version")).get("version")
        except ProviderError:
            return None

    async def _show(self, name: str) -> dict[str, Any]:
        try:
            return await self._request("POST", f"{self.root}/api/show", {"model": name}) or {}
        except ProviderError:
            return {}

    async def _loaded(self) -> dict[str, dict[str, Any]]:
        try:
            payload = await self._request("GET", f"{self.root}/api/ps")
        except ProviderError:
            return {}
        return {item.get("name"): item for item in (payload or {}).get("models", [])}

    async def list_models(self) -> list[ModelInfo]:
        tags = await self._request("GET", f"{self.root}/api/tags")
        names = [item.get("name") for item in (tags or {}).get("models", []) if item.get("name")]
        details = {item.get("name"): item.get("details") or {} for item in (tags or {}).get("models", [])}

        semaphore = asyncio.Semaphore(_SHOW_CONCURRENCY)

        async def show(name):
            async with semaphore:
                return await self._show(name)

        shows, loaded, version = await asyncio.gather(
            asyncio.gather(*(show(name) for name in names)), self._loaded(), self.version())
        # Ollama has accepted JSON-schema `format` (structured outputs) since 0.5
        structured = True if _version_tuple(version) >= (0, 5) else None

        models = []
        for name, show_data in zip(names, shows):
            capabilities = show_data.get("capabilities")
            model_info = show_data.get("model_info") or {}
            context_length = next((value for key, value in model_info.items()
                                   if key.endswith(".context_length") and isinstance(value, int)), None)
            caps = ModelCapabilities(
                tools=("tools" in capabilities) if capabilities is not None else None,
                vision=("vision" in capabilities) if capabilities is not None else None,
                structured_output=structured,
                context_length=context_length,
            )
            is_embedding = capabilities is not None and "embedding" in capabilities and "completion" not in capabilities
            running = loaded.get(name)
            models.append(ModelInfo(
                id=name,
                provider=self.kind,
                type="embedding" if is_embedding else ("vlm" if caps.vision else "llm"),
                capabilities=caps,
                loaded=running is not None,
                details={
                    "family": details.get(name, {}).get("family"),
                    "parameter_size": details.get(name, {}).get("parameter_size"),
                    "quantization": details.get(name, {}).get("quantization_level"),
                    # Context the running instance was actually loaded with (newer Ollama only)
                    "loaded_context_length": (running or {}).get("context_length"),
                },
            ))
        return models

    def warnings(self, models: list[ModelInfo]) -> list[str]:
        small = sorted(
            model.id for model in models
            if isinstance(model.details.get("loaded_context_length"), int)
            and model.details["loaded_context_length"] < MIN_WORKFLOW_CONTEXT
        )
        if not small:
            return []
        return [
            f"{', '.join(small)} {'is' if len(small) == 1 else 'are'} running with less than "
            f"{MIN_WORKFLOW_CONTEXT} tokens of context, which is too small for editing workflows. "
            "Set OLLAMA_CONTEXT_LENGTH (for example 32768) and restart Ollama."
        ]
