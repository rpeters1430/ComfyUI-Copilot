"""Helpers for turning an aiohttp request into the request-scoped LLM config."""

import json
from typing import Optional

from .auth_utils import extract_api_key
from .globals import apply_llm_env_defaults


def get_llm_config_from_headers(request):
    """Extract LLM-related configuration from request headers."""
    return {
        "openai_api_key": request.headers.get('Openai-Api-Key'),
        "openai_base_url": request.headers.get('Openai-Base-Url'),
        # Workflow LLM settings (optional, used by tools/agents that need a different LLM)
        "workflow_llm_api_key": request.headers.get('Workflow-LLM-Api-Key'),
        "workflow_llm_base_url": request.headers.get('Workflow-LLM-Base-Url'),
        "workflow_llm_model": request.headers.get('Workflow-LLM-Model'),
        # Optional model that describes images for chat models without vision
        "vision_llm_model": request.headers.get('Vision-LLM-Model'),
        "vision_llm_base_url": request.headers.get('Vision-LLM-Base-Url'),
        "vision_llm_api_key": request.headers.get('Vision-LLM-Api-Key'),
        # Provider kind (openai / ollama / lmstudio); detected from the URL when absent
        "llm_provider": request.headers.get('Llm-Provider'),
        # auto (decide from model capabilities), agent (tool calling) or json (no tools)
        "tool_mode": request.headers.get('Copilot-Tool-Mode'),
    }


def build_request_config(request, session_id, **extra):
    """Request-scoped LLM config: request headers/body > .env > code defaults."""
    config = {
        "session_id": session_id,
        "copilot_api_key": extract_api_key(request),
        **get_llm_config_from_headers(request),
        **extra,
    }
    return apply_llm_env_defaults(config)


def get_model_select(ext) -> Optional[str]:
    for item in ext or []:
        if isinstance(item, dict) and item.get('type') == 'model_select' and item.get('data'):
            return item['data'][0]
    return None


async def read_json(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None
