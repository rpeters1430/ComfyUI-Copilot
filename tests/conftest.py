"""Test harness: stubs the ComfyUI modules the backend imports at load time.

The backend only runs inside ComfyUI, which provides the `server` and
`folder_paths` modules. These stubs let the `backend` package import on its own.
"""

import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_MODELS_DIR = os.path.realpath(tempfile.mkdtemp(prefix="copilot-models-"))


class _Routes:
    """Mimics aiohttp RouteTableDef decorators; records handlers by (method, path)."""

    def __init__(self):
        self.handlers = {}

    def _register(self, method):
        def decorator_factory(path):
            def decorator(fn):
                self.handlers[(method, path)] = fn
                return fn
            return decorator
        return decorator_factory

    def __getattr__(self, method):
        if method in ("get", "post", "put", "patch", "delete"):
            return self._register(method.upper())
        raise AttributeError(method)


def _install_stubs():
    for name in ("nodes", "execution"):
        sys.modules.setdefault(name, types.ModuleType(name))

    if "server" not in sys.modules:
        server = types.ModuleType("server")
        instance = types.SimpleNamespace(routes=_Routes(), address="127.0.0.1", port=8188, app=None)
        server.PromptServer = types.SimpleNamespace(instance=instance)
        sys.modules["server"] = server

    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = _MODELS_DIR
        folder_paths.folder_names_and_paths = {
            "checkpoints": ([os.path.join(_MODELS_DIR, "checkpoints")], {".safetensors"}),
        }

        def get_folder_paths(name):
            return list(folder_paths.folder_names_and_paths.get(name, ([], set()))[0])

        folder_paths.get_folder_paths = get_folder_paths
        folder_paths.__file__ = os.path.join(_MODELS_DIR, "folder_paths.py")
        sys.modules["folder_paths"] = folder_paths


_install_stubs()

import pytest  # noqa: E402


@pytest.fixture
def models_dir():
    return _MODELS_DIR


@pytest.fixture
def routes():
    # Importing the controllers registers their handlers on the stub PromptServer
    from backend.controller import conversation_api, expert_api, llm_api, provider_api  # noqa: F401

    return sys.modules["server"].PromptServer.instance.routes


@pytest.fixture
def temp_session_db(tmp_path, monkeypatch):
    """Point the session-message DAO at a throwaway SQLite file."""
    from backend.dao import session_message_table

    manager = session_message_table.SessionMessageManager(db_path=str(tmp_path / "session_message.db"))
    monkeypatch.setattr(session_message_table, "session_message_manager", manager)
    return manager



@pytest.fixture
def temp_workflow_db(tmp_path, monkeypatch):
    """Point the workflow-checkpoint DAO at a throwaway SQLite file."""
    from backend.dao import workflow_table

    manager = workflow_table.DatabaseManager(db_path=str(tmp_path / "workflow_debug.db"))
    monkeypatch.setattr(workflow_table, "db_manager", manager)
    return workflow_table
