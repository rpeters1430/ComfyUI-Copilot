"""Resolve which LLM endpoint, key and model a request should use.

Kept free of the openai-agents dependency so routes that only need a plain
chat completion keep working when the agents SDK is missing.
"""

import re
from urllib.parse import urlparse

from .globals import (
    DEFAULT_CHAT_MODEL,
    LLM_DEFAULT_BASE_URL,
    TOKEN_LIMIT_PARAM,
    get_comfyui_copilot_api_key,
    is_lmstudio_url,
)


class LLMConfigurationError(ValueError):
    """Raised when no usable model/endpoint can be resolved."""


def resolve_llm_settings(config: dict | None, model: str | None = None) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model_name).

    Precedence for the model: the request's model_select, then the call-site
    ``model``, then the CC_DEFAULT_MODEL env var.
    """
    config = config or {}
    base_url = config.get("openai_base_url") or LLM_DEFAULT_BASE_URL
    api_key = config.get("openai_api_key") or get_comfyui_copilot_api_key() or ""

    # The OpenAI client refuses an empty key, but local servers (LMStudio, Ollama, llama.cpp,
    # vLLM) usually need none. A placeholder lets those work; real endpoints answer 401.
    if not api_key:
        api_key = "lmstudio-local" if is_lmstudio_url(base_url) else "no-key-set"

    model_name = config.get("model_select") or model or DEFAULT_CHAT_MODEL
    if not model_name:
        raise LLMConfigurationError(
            "No model selected. Choose a model in the Copilot settings, or set "
            "CC_DEFAULT_MODEL in ComfyUI-Copilot/.env."
        )
    return base_url, api_key, model_name


# OpenAI's reasoning models reject max_tokens and require max_completion_tokens
_COMPLETION_TOKENS_MODELS = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)


def token_limit_param(model_name: str, base_url: str) -> str:
    """Name of the output-token-limit parameter this endpoint/model accepts."""
    if TOKEN_LIMIT_PARAM in ("max_tokens", "max_completion_tokens"):
        return TOKEN_LIMIT_PARAM
    host = (urlparse(base_url or "").hostname or "").lower()
    # Strip provider prefixes like "openai/gpt-5" (OpenRouter style)
    bare_model = (model_name or "").rsplit("/", 1)[-1]
    if host == "api.openai.com" or _COMPLETION_TOKENS_MODELS.match(bare_model):
        return "max_completion_tokens"
    return "max_tokens"


def token_limit_kwargs(model_name: str, base_url: str, limit: int) -> dict:
    return {token_limit_param(model_name, base_url): limit}
