import pytest

from backend import agent_factory
from backend.agent_factory import LLMConfigurationError, resolve_llm_settings
from backend.utils import globals as copilot_globals
from backend.utils.key_utils import redact_config, workflow_config_adapt
from backend.utils.request_context import RewriteContext, set_request_context


def test_workflow_config_keeps_chat_model_when_no_workflow_model():
    config = {"model_select": "gpt-4.1-mini", "openai_base_url": "https://example.test/v1"}
    adapted = workflow_config_adapt(config)
    assert adapted["model_select"] == "gpt-4.1-mini"
    assert config == {"model_select": "gpt-4.1-mini", "openai_base_url": "https://example.test/v1"}


def test_workflow_config_prefers_workflow_llm_settings():
    adapted = workflow_config_adapt({
        "model_select": "chat-model",
        "openai_api_key": "chat-key",
        "workflow_llm_model": "workflow-model",
        "workflow_llm_api_key": "workflow-key",
        "workflow_llm_base_url": "https://workflow.test/v1",
    })
    assert adapted["model_select"] == "workflow-model"
    assert adapted["openai_api_key"] == "workflow-key"
    assert adapted["openai_base_url"] == "https://workflow.test/v1"
    assert adapted["workflow_llm_api_key"] is None


def test_redact_config_hides_every_key():
    redacted = redact_config({"openai_api_key": "sk-1", "workflow_llm_api_key": "sk-2",
                              "copilot_api_key": "sk-3", "model_select": "m"})
    assert redacted == {"openai_api_key": "***", "workflow_llm_api_key": "***",
                        "copilot_api_key": "***", "model_select": "m"}


def test_resolve_llm_settings_precedence(monkeypatch):
    monkeypatch.setattr(agent_factory, "DEFAULT_CHAT_MODEL", "env-default")
    _, _, model = resolve_llm_settings({"model_select": "selected"}, "call-site")
    assert model == "selected"
    _, _, model = resolve_llm_settings({}, "call-site")
    assert model == "call-site"
    _, _, model = resolve_llm_settings({}, None)
    assert model == "env-default"


def test_resolve_llm_settings_without_model_raises(monkeypatch):
    monkeypatch.setattr(agent_factory, "DEFAULT_CHAT_MODEL", None)
    with pytest.raises(LLMConfigurationError):
        resolve_llm_settings({}, None)


def test_resolve_llm_settings_lmstudio_needs_no_key():
    base_url, api_key, _ = resolve_llm_settings(
        {"openai_base_url": "http://localhost:1234/v1", "model_select": "local"}, None)
    assert base_url == "http://localhost:1234/v1"
    assert api_key == "lmstudio-local"


def test_copilot_api_key_is_request_scoped():
    import asyncio

    async def request(key):
        set_request_context("s", None, {"copilot_api_key": key})
        await asyncio.sleep(0)
        return copilot_globals.get_comfyui_copilot_api_key()

    async def main():
        return await asyncio.gather(request("key-a"), request("key-b"))

    assert asyncio.run(main()) == ["key-a", "key-b"]


def test_env_flag_parses_false_strings(monkeypatch):
    monkeypatch.setenv("COPILOT_TEST_FLAG", "false")
    assert copilot_globals._env_flag("COPILOT_TEST_FLAG") is False
    monkeypatch.setenv("COPILOT_TEST_FLAG", "1")
    assert copilot_globals._env_flag("COPILOT_TEST_FLAG") is True


def test_rewrite_context_node_infos_is_writable():
    context = RewriteContext()
    context.node_infos["KSampler"] = "{}"
    assert RewriteContext().node_infos == {}
