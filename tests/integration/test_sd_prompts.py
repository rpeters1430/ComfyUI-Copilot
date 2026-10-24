import asyncio
import json

import pytest

from backend.controller.llm_api import parse_prompt_list
from tests.fixtures.fake_llm import copilot_with_fake_llm


def _generate(routes, body, **reply):
    async def run():
        async with copilot_with_fake_llm(routes.handlers, **reply) as (client, base_url, llm_requests):
            response = await client.post("/api/param_debug/generate_sd_prompts", json=body,
                                         headers={"Openai-Base-Url": base_url})
            return response.status, await response.json(), llm_requests
    return asyncio.run(run())


def test_generates_prompts_with_the_users_llm(routes):
    reply = "```json\n" + json.dumps(["a cat, studio light", "a cat, watercolor"]) + "\n```"
    status, result, llm_requests = _generate(routes, {"text": "a cat", "model_select": "fake-model"}, text=reply)

    assert status == 200
    assert result == {"success": True, "data": ["a cat, studio light", "a cat, watercolor"]}
    assert llm_requests[0]["body"]["model"] == "fake-model"


def test_reports_llm_errors(routes):
    _, result, _ = _generate(routes, {"text": "a cat", "model_select": "fake-model"}, status=401, error="bad key")
    assert result["success"] is False
    assert "HTTP 401" in result["message"]


def test_requires_text(routes):
    status, _, _ = _generate(routes, {"text": "  "})
    assert status == 400


@pytest.mark.parametrize("reply,expected", [
    ('["one", "two"]', ["one", "two"]),
    ('Here you go:\n["one", " ", "two"]\nEnjoy', ["one", "two"]),
    ("1. one\n2. two", ["one", "two"]),
])
def test_parse_prompt_list(reply, expected):
    assert parse_prompt_list(reply) == expected
