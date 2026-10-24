import json
import os

import pytest

from backend.controller import conversation_api
from backend.controller.conversation_api import get_model_select, resolve_model_dest_dir


class _Request:
    def __init__(self, body=None, headers=None, raw=None):
        self._body = body
        self._raw = raw
        self.headers = headers or {}

    async def json(self):
        if self._raw is not None:
            return json.loads(self._raw)
        return self._body


def test_dest_dir_inside_models_dir(models_dir):
    assert resolve_model_dest_dir("loras/sdxl", "loras") == os.path.join(models_dir, "loras", "sdxl")


@pytest.mark.parametrize("dest_dir", ["../outside", "../../etc", "loras/../../outside"])
def test_dest_dir_traversal_rejected(dest_dir):
    with pytest.raises(ValueError):
        resolve_model_dest_dir(dest_dir, "loras")


def test_dest_dir_absolute_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_model_dest_dir(str(tmp_path), "loras")


def test_model_type_traversal_rejected():
    with pytest.raises(ValueError):
        resolve_model_dest_dir(None, "../../outside")


def test_configured_folder_used_for_known_type(models_dir):
    assert resolve_model_dest_dir(None, "checkpoints") == os.path.join(models_dir, "checkpoints")


def test_get_model_select_tolerates_missing_ext():
    assert get_model_select(None) is None
    assert get_model_select([{"type": "other"}, {"type": "model_select", "data": ["m1"]}]) == "m1"
    assert get_model_select([{"type": "model_select", "data": []}]) is None


def test_build_request_config_reads_bearer_key():
    request = _Request(headers={"Authorization": "Bearer abc", "Openai-Api-Key": "sk-x"})
    config = conversation_api.build_request_config(request, "s1", model_select="m")
    assert config["copilot_api_key"] == "abc"
    assert config["openai_api_key"] == "sk-x"
    assert config["model_select"] == "m"


@pytest.mark.parametrize("routes_key", [
    ("POST", "/api/chat/invoke"),
    ("POST", "/api/debug-agent"),
    ("POST", "/api/download-model"),
    ("POST", "/api/save-workflow-checkpoint"),
])
def test_invalid_json_returns_400(routes, routes_key):
    import asyncio

    handler = routes.handlers[routes_key]
    response = asyncio.run(handler(_Request(raw="{not json")))
    assert response.status == 400


def test_download_model_rejects_traversal(routes):
    import asyncio

    handler = routes.handlers[("POST", "/api/download-model")]
    response = asyncio.run(handler(_Request(body={
        "id": "1", "model_id": "org/model", "model_type": "loras", "dest_dir": "../../outside",
    })))
    assert response.status == 400
