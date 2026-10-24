"""Turn LLM client exceptions into messages a user can act on."""

from typing import Optional

import openai


def describe_llm_error(error: BaseException, base_url: Optional[str] = None, model: Optional[str] = None) -> str:
    where = f" at {base_url}" if base_url else ""
    model_name = f"'{model}'" if model else "the selected model"

    if isinstance(error, openai.AuthenticationError):
        return (f"The LLM endpoint{where} rejected the API key (HTTP 401). "
                "Check the API key in the Copilot settings.")
    if isinstance(error, openai.PermissionDeniedError):
        return (f"The API key is not allowed to use {model_name}{where} (HTTP 403). "
                "Check the key's permissions or choose another model.")
    if isinstance(error, openai.NotFoundError):
        return (f"HTTP 404 from the LLM endpoint{where}: {model_name} does not exist there, or the base URL "
                "is wrong (OpenAI-compatible base URLs usually end in /v1).")
    if isinstance(error, openai.APITimeoutError):
        return f"The LLM endpoint{where} timed out. Try again, or pick a faster model."
    if isinstance(error, openai.APIConnectionError):
        return f"Could not connect to the LLM endpoint{where}. Make sure the server is running and the base URL is correct."
    if isinstance(error, openai.BadRequestError) and "max_tokens" in str(error):
        return (f"{model_name} rejected the 'max_tokens' parameter. Set CC_TOKEN_LIMIT_PARAM=max_completion_tokens "
                f"in ComfyUI-Copilot/.env. Details: {error}")
    return str(error)
