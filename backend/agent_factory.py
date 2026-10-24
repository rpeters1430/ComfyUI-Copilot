'''
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-07-31 19:38:08
LastEditors: ai-business-hql ai.bussiness.hql@gmail.com
LastEditTime: 2026-01-12 11:11:53
FilePath: /comfyui_copilot/backend/agent_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

try:
    from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, Runner, set_tracing_disabled, set_default_openai_api
    if not hasattr(__import__('agents'), 'Agent'):
        raise ImportError
except Exception:
    # Give actionable guidance without crashing obscurely
    raise ImportError(
        "Detected incorrect or missing 'agents' package. "
        "Please uninstall legacy RL 'agents' (and tensorflow/gym if pulled transitively) and install openai-agents. "
        "Commands:\n"
        "  python -m pip uninstall -y agents gym tensorflow\n"
        "  python -m pip install -U openai-agents\n\n"
        "Alternatively, keep both by setting COMFYUI_COPILOT_PREFER_OPENAI_AGENTS=1 so this plugin prefers openai-agents."
    )
from .utils.globals import (
    LLM_DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    ENABLE_TRACING,
    get_comfyui_copilot_api_key,
    is_lmstudio_url,
)
from openai import AsyncOpenAI

set_default_openai_api("chat_completions")
set_tracing_disabled(not ENABLE_TRACING)


class LLMConfigurationError(ValueError):
    """Raised when no usable model/endpoint can be resolved for an agent."""


def resolve_llm_settings(config: dict | None, model: str | None = None) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model_name) for an agent.

    Precedence for the model: the request's model_select, then the call-site
    ``model``, then the CC_DEFAULT_MODEL env var.
    """
    config = config or {}
    base_url = config.get("openai_base_url") or LLM_DEFAULT_BASE_URL
    api_key = config.get("openai_api_key") or get_comfyui_copilot_api_key() or ""

    # LMStudio typically doesn't require an API key, use a placeholder
    if not api_key and is_lmstudio_url(base_url):
        api_key = "lmstudio-local"

    model_name = config.get("model_select") or model or DEFAULT_CHAT_MODEL
    if not model_name:
        raise LLMConfigurationError(
            "No model selected. Choose a model in the Copilot settings, or set "
            "CC_DEFAULT_MODEL in ComfyUI-Copilot/.env."
        )
    return base_url, api_key, model_name


def create_agent(**kwargs) -> Agent:
    # 通过用户配置拿/环境变量
    config = kwargs.pop("config", None) or {}
    session_id = config.get("session_id")
    default_headers = {"X-Session-ID": session_id} if session_id else {}

    base_url, api_key, model_name = resolve_llm_settings(config, kwargs.pop("model", None))

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )
    model = OpenAIChatCompletionsModel(model_name, openai_client=client)

    if config.get("max_tokens"):
        return Agent(model=model, model_settings=ModelSettings(max_tokens=config["max_tokens"]), **kwargs)
    return Agent(model=model, **kwargs)
