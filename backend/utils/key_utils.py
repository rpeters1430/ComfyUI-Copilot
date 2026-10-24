from copy import deepcopy

SENSITIVE_CONFIG_KEYS = ("openai_api_key", "workflow_llm_api_key", "vision_llm_api_key", "copilot_api_key")


def workflow_config_adapt(config: dict) -> dict:
    """Return a deep-copied and adapted workflow config without mutating input.

    - Map workflow_llm_api_key -> openai_api_key (and null out original key)
    - Map workflow_llm_base_url -> openai_base_url (and null out original key)
    - Map workflow_llm_model -> model_select. When no workflow model is configured
      the chat model the user selected is kept, so workflow agents run on the same
      endpoint/model as chat instead of a model the endpoint may not serve.
    """
    if not config:
        return {}

    new_config = deepcopy(config)

    if new_config.get("workflow_llm_api_key"):
        new_config["openai_api_key"] = new_config.get("workflow_llm_api_key")
        new_config["workflow_llm_api_key"] = None
    if new_config.get("workflow_llm_base_url"):
        new_config["openai_base_url"] = new_config.get("workflow_llm_base_url")
        new_config["workflow_llm_base_url"] = None
    if new_config.get("workflow_llm_model"):
        new_config["model_select"] = new_config.get("workflow_llm_model")
        new_config["workflow_llm_model"] = None

    return new_config


def redact_config(config: dict) -> dict:
    """Copy of a request config that is safe to log."""
    redacted = dict(config or {})
    for key in SENSITIVE_CONFIG_KEYS:
        if redacted.get(key):
            redacted[key] = "***"
    return redacted
