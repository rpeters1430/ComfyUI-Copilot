"""Chat adapts to model capabilities: JSON mode without tools, image handling without vision."""

import asyncio
import json

import pytest

from backend.providers import registry
from backend.service import mcp_client, workflow_rewrite_tools
from tests.fixtures.fake_llm import copilot_with_fake_llm, lmstudio_model, read_chat_lines

WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
    "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1, "steps": 20}},
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, temp_session_db, temp_workflow_db):
    monkeypatch.setattr(mcp_client, "REMOTE_MCP_URL", None)
    monkeypatch.setattr(mcp_client, "SEARCH_MCP_URL", None)

    async def no_comfyui():
        return None  # don't reach for a real ComfyUI's node list

    monkeypatch.setattr(workflow_rewrite_tools, "node_definitions", no_comfyui)
    registry.clear_capability_cache()


def _chat(routes, messages, headers=None, model="fake-model", **reply):
    async def run():
        async with copilot_with_fake_llm(routes.handlers, **reply) as (client, base_url, llm_requests):
            response = await client.post("/api/chat/invoke", json={
                "session_id": "neg-session",
                "messages": messages,
                "ext": [{"type": "model_select", "data": [model]}],
            }, headers={"Openai-Base-Url": base_url, **(headers or {})})
            return await read_chat_lines(response), llm_requests
    return asyncio.run(run())


def _decision(action, reply, intent=""):
    return json.dumps({"action": action, "reply": reply, "rewrite_intent": intent})


def _chat_requests(llm_requests):
    return [request["body"] for request in llm_requests]


def test_json_mode_answers_without_tools(routes):
    lines, llm_requests = _chat(routes, [{"role": "user", "content": "what is a KSampler?"}],
                                headers={"Copilot-Tool-Mode": "json"},
                                texts=[_decision("answer", "It samples latents.")])

    assert lines[-1]["finished"] is True
    assert lines[-1]["text"] == "It samples latents."
    assert all("tools" not in body for body in _chat_requests(llm_requests))


def test_auto_mode_uses_json_for_model_without_tool_calling(routes):
    lines, llm_requests = _chat(routes, [{"role": "user", "content": "hi"}],
                                headers={"Llm-Provider": "lmstudio"},
                                models=[lmstudio_model("fake-model", tools=False)],
                                texts=[_decision("answer", "Hello!")])

    assert "doesn't support tool calling" in lines[-1]["text"]
    assert lines[-1]["text"].endswith("Hello!")
    assert all("tools" not in body for body in _chat_requests(llm_requests))


def test_auto_mode_keeps_agent_for_tool_capable_model(routes):
    lines, llm_requests = _chat(routes, [{"role": "user", "content": "hi"}],
                                headers={"Llm-Provider": "lmstudio"},
                                models=[lmstudio_model("fake-model", tools=True)], text="Agent reply")

    assert lines[-1]["text"] == "Agent reply"
    assert "tools" in _chat_requests(llm_requests)[0]


def test_json_mode_rewrite_applies_valid_workflow(routes, temp_workflow_db):
    temp_workflow_db.save_workflow_data("neg-session", WORKFLOW)
    new_workflow = {**WORKFLOW, "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0]}}}

    lines, _ = _chat(routes, [{"role": "user", "content": "add a VAE decode"}],
                     headers={"Copilot-Tool-Mode": "json"},
                     texts=[_decision("rewrite_workflow", "Adding a VAE Decode.", "Add VAEDecode after node 2"),
                            json.dumps({"workflow_data": new_workflow})])

    final = lines[-1]
    assert final["finished"] is True
    assert "has been updated" in final["text"]
    assert final["ext"][0]["type"] == "workflow_update"
    assert final["ext"][0]["data"]["workflow_data"] == new_workflow
    assert temp_workflow_db.get_workflow_data("neg-session") == new_workflow


def test_json_mode_rewrite_rejects_invalid_workflow(routes, temp_workflow_db):
    temp_workflow_db.save_workflow_data("neg-session", WORKFLOW)
    broken = {"1": WORKFLOW["1"], "2": {"class_type": "KSampler", "inputs": {"model": ["99", 0]}}}

    lines, llm_requests = _chat(routes, [{"role": "user", "content": "change the sampler"}],
                                headers={"Copilot-Tool-Mode": "json"},
                                texts=[_decision("rewrite_workflow", "Changing it.", "Use another sampler"),
                                       json.dumps({"workflow_data": broken}), json.dumps({"workflow_data": broken})])

    final = lines[-1]
    # One decision call, then the rewrite plus one corrective retry
    assert len(llm_requests) == 3
    assert "A previous attempt was rejected" in json.dumps(llm_requests[2]["body"]["messages"])
    assert "not applied" in final["text"]
    assert "node 99" in final["text"]
    assert final["ext"] is None
    # The saved workflow (and so the canvas) is untouched
    assert temp_workflow_db.get_workflow_data("neg-session") == WORKFLOW


def test_json_mode_retry_repairs_rejected_rewrite(routes, temp_workflow_db):
    temp_workflow_db.save_workflow_data("neg-session", WORKFLOW)
    broken = {"1": WORKFLOW["1"], "2": {"class_type": "KSampler", "inputs": {"model": ["99", 0]}}}
    fixed = {**WORKFLOW, "2": {**WORKFLOW["2"], "inputs": {**WORKFLOW["2"]["inputs"], "steps": 30}}}

    lines, _ = _chat(routes, [{"role": "user", "content": "use 30 steps"}],
                     headers={"Copilot-Tool-Mode": "json"},
                     texts=[_decision("rewrite_workflow", "Setting 30 steps.", "steps=30"),
                            json.dumps({"workflow_data": broken}), json.dumps({"workflow_data": fixed})])

    assert "has been updated" in lines[-1]["text"]
    assert temp_workflow_db.get_workflow_data("neg-session") == fixed


IMAGE_MESSAGE = [{"role": "user", "content": [
    {"type": "input_text", "text": "make something like this"},
    {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,aGk="},
]}]


def test_images_are_described_for_a_model_without_vision(routes):
    lines, llm_requests = _chat(
        routes, IMAGE_MESSAGE,
        headers={"Llm-Provider": "lmstudio", "Vision-LLM-Model": "vision-model"},
        models=[lmstudio_model("fake-model", tools=True, vision=False), lmstudio_model("vision-model", vision=True)],
        texts=["A red fox in snow."], text="Here is a plan.")

    describe, chat = _chat_requests(llm_requests)
    assert describe["model"] == "vision-model"
    assert describe["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,aGk="
    chat_text = json.dumps(chat["messages"])
    assert "A red fox in snow." in chat_text
    assert "data:image/png" not in chat_text
    assert lines[-1]["text"] == "Here is a plan."


def test_images_dropped_with_note_without_vision_model(routes):
    _, llm_requests = _chat(routes, IMAGE_MESSAGE, headers={"Llm-Provider": "lmstudio"},
                            models=[lmstudio_model("fake-model", tools=True, vision=False)], text="ok")

    chat_text = json.dumps(_chat_requests(llm_requests)[0]["messages"])
    assert "cannot see images" in chat_text
    assert "data:image/png" not in chat_text


def test_images_kept_for_vision_model(routes):
    _, llm_requests = _chat(routes, IMAGE_MESSAGE, headers={"Llm-Provider": "lmstudio"},
                            models=[lmstudio_model("fake-model", tools=True, vision=True)], text="ok")

    assert "data:image/png" in json.dumps(_chat_requests(llm_requests)[0]["messages"])
