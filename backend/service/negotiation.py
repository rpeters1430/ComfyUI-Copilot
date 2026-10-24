"""Adapt a chat request to what the selected model can actually do.

- Models without tool calling can't drive the agent router; they get the
  structured-JSON fallback instead (see json_fallback.py).
- Models without vision can't take images; a configured vision model
  describes them first, otherwise they're dropped with a note.
"""

from typing import Any, Optional

from openai import AsyncOpenAI

from ..providers import ModelInfo, lookup_model
from ..utils.globals import LLM_DEFAULT_BASE_URL
from ..utils.llm_settings import resolve_llm_settings
from ..utils.logger import log

TOOL_MODES = ("auto", "agent", "json")

VISION_PROMPT = (
    "Describe this image for someone building an image-generation workflow: subject, setting, style, "
    "composition, colors, lighting and any visible text. Be specific and concise."
)

_TEXT_TYPES = ("text", "input_text")
_IMAGE_TYPES = ("image_url", "input_image")


async def chat_model_info(config: dict) -> Optional[ModelInfo]:
    return await lookup_model(
        config.get("openai_base_url") or LLM_DEFAULT_BASE_URL,
        config.get("openai_api_key"),
        config.get("model_select"),
        config.get("llm_provider"),
    )


def resolve_tool_mode(config: dict, model: Optional[ModelInfo]) -> str:
    """'agent' (tool-calling router) or 'json' (structured fallback)."""
    mode = (config.get("tool_mode") or "auto").lower()
    if mode in ("agent", "json"):
        return mode
    # Unknown capability: try the agent; only a provider that says "no tools" switches modes
    if model is not None and model.capabilities.tools is False:
        return "json"
    return "agent"


def image_url_of(part: dict) -> Optional[str]:
    url = part.get("image_url")
    return url.get("url") if isinstance(url, dict) else url


def has_images(messages: list[dict]) -> bool:
    return any(isinstance(part, dict) and part.get("type") in _IMAGE_TYPES
               for message in messages if isinstance(message.get("content"), list)
               for part in message["content"])


def to_chat_messages(messages: list[dict]) -> list[dict]:
    """Convert Responses-style content parts (input_text/input_image) to chat-completions parts."""
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in _TEXT_TYPES:
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") in _IMAGE_TYPES and image_url_of(part):
                    parts.append({"type": "image_url", "image_url": {"url": image_url_of(part)}})
            content = parts
        converted.append({"role": message.get("role", "user"), "content": content})
    return converted


async def describe_image(client: AsyncOpenAI, model: str, url: str) -> str:
    completion = await client.chat.completions.create(model=model, messages=[{
        "role": "user",
        "content": [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": url}}],
    }])
    return (completion.choices[0].message.content or "").strip()


async def prepare_images(messages: list[dict], config: dict, model: Optional[ModelInfo]) -> list[dict]:
    """Replace images with text when the chat model can't see them."""
    if not has_images(messages) or model is None or model.capabilities.vision is not False:
        return messages

    vision_model = config.get("vision_llm_model")
    client = None
    if vision_model:
        base_url, api_key, _ = resolve_llm_settings({
            "openai_base_url": config.get("vision_llm_base_url") or config.get("openai_base_url"),
            "openai_api_key": config.get("vision_llm_api_key") or config.get("openai_api_key"),
            "model_select": vision_model,
        })
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)

    result, image_number = [], 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            result.append(message)
            continue
        parts = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") in _IMAGE_TYPES):
                parts.append(part)
                continue
            image_number += 1
            text_type = "input_text" if part.get("type") == "input_image" else "text"
            parts.append({"type": text_type, "text": await _image_as_text(client, vision_model, part,
                                                                          image_number, model.id)})
        result.append({**message, "content": parts})
    return result


async def _image_as_text(client: Optional[AsyncOpenAI], vision_model: Optional[str], part: dict,
                         number: int, chat_model: str) -> str:
    if client is None:
        return (f"[Image {number} was attached, but {chat_model} cannot see images. "
                "Choose a vision model in the Copilot settings to have images described.]")
    try:
        description = await describe_image(client, vision_model, image_url_of(part))
        return f"[Image {number}, described by {vision_model}: {description}]"
    except Exception as e:
        log.error(f"Vision model {vision_model} failed: {e}")
        return f"[Image {number} could not be described by {vision_model}: {e}]"


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content: Any = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") in _TEXT_TYPES)
    return ""
