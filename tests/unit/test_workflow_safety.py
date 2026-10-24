import types

import httpx
import openai
import pytest
from pydantic import BaseModel

from backend.utils.structured_output import complete_json, extract_json_object
from backend.utils.workflow_validation import validate_api_workflow

VALID = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 5, "denoise": 1.0}},
}


DEFINITIONS = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"]]}}},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT"]},
                           "optional": {"denoise": ["FLOAT"]}, "hidden": {"unique_id": "UNIQUE_ID"}}},
}


def test_valid_workflow_passes():
    assert validate_api_workflow(VALID) == []
    assert validate_api_workflow(VALID, DEFINITIONS) == []


@pytest.mark.parametrize("workflow,fragment", [
    ({}, "non-empty"),
    ([], "non-empty"),
    ({"1": "x"}, "not an object"),
    ({"1": {"inputs": {}}}, "no class_type"),
    ({"1": {"class_type": "A", "inputs": []}}, "not an object"),
    ({"1": {"class_type": "A", "inputs": {"x": ["7", 0]}}}, "node 7"),
])
def test_invalid_workflows_are_rejected(workflow, fragment):
    problems = validate_api_workflow(workflow)
    assert problems and fragment in " ".join(problems)


def test_uninstalled_node_type_is_reported():
    problems = validate_api_workflow(VALID, {"CheckpointLoaderSimple": DEFINITIONS["CheckpointLoaderSimple"]})
    assert problems == ["Node 2 uses node type 'KSampler', which is not installed."]


def test_renamed_input_is_caught():
    # What a 7B model actually did when asked to change the CFG: renamed "cfg" to "cfg_scale"
    renamed = {"1": VALID["1"], "2": {"class_type": "KSampler",
                                      "inputs": {"model": ["1", 0], "seed": 5, "steps_count": 30}}}
    definitions = {**DEFINITIONS, "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT"],
                                                                      "steps": ["INT"]}}}}
    assert validate_api_workflow(renamed, definitions) == [
        "Node 2 (KSampler) is missing required input 'steps'.",
        "Node 2 (KSampler) has no input named 'steps_count'.",
    ]


def test_scalar_widget_values_pass():
    workflow = {"1": {"class_type": "A", "inputs": {"size": 512, "text": "a [1, 0] cat", "on": True}}}
    assert validate_api_workflow(workflow) == []


def test_two_element_lists_are_links_like_in_comfyui():
    problems = validate_api_workflow({"1": {"class_type": "A", "inputs": {"size": [512, 512]}}})
    assert problems == ["Node 1 input 'size' links to node 512, which does not exist."]


def test_extract_json_object_skips_prose_and_fences():
    assert extract_json_object('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        extract_json_object("no json here {")


class Answer(BaseModel):
    value: int


class _FakeCompletions:
    """parse() is rejected like a server without json_schema support; create() records its arguments."""

    def __init__(self, replies, reject_json_mode=False):
        self.replies = list(replies)
        self.reject_json_mode = reject_json_mode
        self.create_calls = []

    def _bad_request(self, message):
        response = httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions"))
        return openai.BadRequestError(message, response=response, body=None)

    def parse(self, **kwargs):
        raise self._bad_request("response_format json_schema is not supported")

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.reject_json_mode and kwargs.get("response_format"):
            raise self._bad_request("json_object not supported")
        message = types.SimpleNamespace(content=self.replies.pop(0))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _client(completions):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def test_complete_json_falls_back_to_json_mode():
    completions = _FakeCompletions(['{"value": 3}'])
    assert complete_json(_client(completions), "m", [{"role": "user", "content": "x"}], Answer).value == 3
    assert completions.create_calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_falls_back_to_plain_text():
    completions = _FakeCompletions(['The answer is {"value": 4}.'], reject_json_mode=True)
    assert complete_json(_client(completions), "m", [], Answer).value == 4
    assert "response_format" not in completions.create_calls[-1]


def test_complete_json_raises_when_no_valid_json():
    completions = _FakeCompletions(['{"value": "nope"}', "still no"])
    with pytest.raises(ValueError):
        complete_json(_client(completions), "m", [], Answer)
