"""Get a schema-shaped JSON answer from any OpenAI-compatible model.

Servers and models differ in what they support: json_schema structured
outputs, plain JSON mode, or neither. This tries them in that order and
validates the result against the Pydantic schema either way.
"""

import json
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .logger import log

T = TypeVar("T", bound=BaseModel)

# Raised by servers or the SDK when a response_format mode isn't supported or the reply doesn't parse
_UNSUPPORTED = (openai.BadRequestError, openai.UnprocessableEntityError, openai.NotFoundError,
                TypeError, ValueError, ValidationError)


def extract_json_object(text: str) -> Any:
    """First JSON object in a model reply, ignoring code fences and surrounding prose."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            return value
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    raise ValueError("The model reply contains no JSON object")


def complete_json(client: openai.OpenAI, model: str, messages: list[dict], schema: type[T], **kwargs) -> T:
    """Chat completion parsed into ``schema``; tries json_schema, json_object, then plain text."""
    try:
        completion = client.chat.completions.parse(model=model, messages=messages, response_format=schema, **kwargs)
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except _UNSUPPORTED as e:
        log.info(f"Structured outputs unavailable for {model} ({type(e).__name__}); falling back to JSON mode")

    schema_hint = {
        "role": "system",
        "content": "Reply with only a JSON object matching this JSON schema:\n"
                   + json.dumps(schema.model_json_schema(), ensure_ascii=False),
    }
    for response_format in ({"type": "json_object"}, None):
        extra = {"response_format": response_format} if response_format else {}
        try:
            completion = client.chat.completions.create(model=model, messages=[schema_hint, *messages],
                                                        **extra, **kwargs)
            return schema.model_validate(extract_json_object(completion.choices[0].message.content or ""))
        except _UNSUPPORTED as e:
            last_error = e
            log.info(f"JSON reply attempt failed for {model} ({type(e).__name__}: {e})")
    raise ValueError(f"{model} did not return valid JSON: {last_error}")
