'''
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-07-31 19:38:08
LastEditors: ai-business-hql ai.bussiness.hql@gmail.com
LastEditTime: 2026-01-12 11:11:53
FilePath: /comfyui_copilot/backend/agent_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

try:
    from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled, set_default_openai_api
except Exception:
    # Give actionable guidance without crashing obscurely
    raise ImportError(
        "Detected incorrect or missing 'agents' package. "
        "Please uninstall legacy RL 'agents' (and tensorflow/gym if pulled transitively) and install openai-agents. "
        "Commands:\n"
        "  python -m pip uninstall -y agents gym tensorflow\n"
        "  python -m pip install -U openai-agents"
    )
from .utils.globals import ENABLE_TRACING
from .utils.llm_settings import (  # noqa: F401  (re-exported for existing callers)
    LLMConfigurationError,
    resolve_llm_settings,
    token_limit_kwargs,
    token_limit_param,
)
from openai import AsyncOpenAI

set_default_openai_api("chat_completions")
set_tracing_disabled(not ENABLE_TRACING)


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
        limit = config["max_tokens"]
        if token_limit_param(model_name, base_url) == "max_completion_tokens":
            settings = ModelSettings(extra_args={"max_completion_tokens": limit})
        else:
            settings = ModelSettings(max_tokens=limit)
        return Agent(model=model, model_settings=settings, **kwargs)
    return Agent(model=model, **kwargs)
