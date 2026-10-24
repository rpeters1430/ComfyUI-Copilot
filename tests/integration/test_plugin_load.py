"""Loads the repo the way ComfyUI loads a custom node, in a fresh interpreter."""

import json
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOADER = textwrap.dedent("""
    import importlib.util, json, os, sys, types
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import conftest  # installs the ComfyUI module stubs
    if BLOCK_AGENTS:
        sys.modules["agents"] = None  # makes `import agents` raise ImportError

    instance = sys.modules["server"].PromptServer.instance
    static = []
    instance.app = types.SimpleNamespace(add_routes=static.extend)

    init = os.path.join(ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location("ComfyUI-Copilot", init, submodule_search_locations=[ROOT])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    conversation_api = sys.modules["ComfyUI-Copilot.backend.controller.conversation_api"]
    print(json.dumps({
        "routes": sorted(f"{m} {p}" for m, p in instance.routes.handlers),
        "static": len(static),
        "web_directory": module.WEB_DIRECTORY,
        "agents_unavailable": bool(conversation_api.AGENTS_UNAVAILABLE),
    }))
""")


def _load_plugin(block_agents):
    code = f"ROOT = {ROOT!r}\nBLOCK_AGENTS = {block_agents}\n" + LOADER
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_plugin_registers_all_routes():
    loaded = _load_plugin(block_agents=False)
    assert loaded["static"] == 1
    assert loaded["web_directory"] == "entry"
    assert not loaded["agents_unavailable"]
    assert "POST /api/chat/invoke" in loaded["routes"]
    assert len(loaded["routes"]) == 20


def test_plugin_loads_without_openai_agents():
    loaded = _load_plugin(block_agents=True)
    # The sidebar UI is still served and every route exists; chat/debug explain the problem
    assert loaded["static"] == 1
    assert loaded["agents_unavailable"]
    for route in ("POST /api/chat/invoke", "POST /api/debug-agent", "GET /api/model_config",
                  "POST /api/save-workflow-checkpoint", "GET /api/expert/experts"):
        assert route in loaded["routes"]
