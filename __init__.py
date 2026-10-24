'''
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-02-17 20:53:45
LastEditors: ai-business-hql ai.bussiness.hql@gmail.com
LastEditTime: 2025-11-20 20:03:20
FilePath: /comfyui_copilot/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.

import sys
import asyncio


# Ensure 'agents' resolves to openai-agents (not legacy RL package)
try:
    import importlib.metadata as _im
    from pathlib import Path as _Path

    try:
        _dist = _im.distribution("openai-agents")
    except _im.PackageNotFoundError:
        _dist = None

    if _dist and _dist.files:
        _init_rel = next((f for f in _dist.files if str(f).replace("\\", "/").endswith("agents/__init__.py")), None)
        if _init_rel:
            _init_path = _dist.locate_file(_init_rel)
            _agents_parent = _Path(_init_path).parent.parent
            _pp = str(_agents_parent)
            if _pp not in sys.path:
                sys.path.insert(0, _pp)

            # If an incompatible 'agents' was already imported, drop it so the correct one can load
            m = sys.modules.get("agents")
            if m is not None and not hasattr(m, "Agent"):
                sys.modules.pop("agents", None)
except Exception:
    # Fail-open: never block plugin loading if aliasing fails
    pass

import os

WEB_DIRECTORY = "entry"
NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']


def _register_with_comfyui():
    import importlib
    import traceback

    import server
    from aiohttp import web

    # Importing a controller registers its routes on PromptServer. Load each one on its own so
    # a broken dependency disables only its routes, and the sidebar UI is still served.
    for controller in ("conversation_api", "llm_api", "expert_api", "provider_api", "discovery_api"):
        try:
            importlib.import_module(f".backend.controller.{controller}", __package__)
        except Exception:
            print(f"[ComfyUI-Copilot] Failed to load {controller}; its routes are unavailable:")
            traceback.print_exc()

    dist_path = os.path.join(os.path.dirname(__file__), 'dist', 'copilot_web')
    if os.path.exists(dist_path):
        server.PromptServer.instance.app.add_routes([
            web.static('/copilot_web/', dist_path),
        ])
    else:
        print(f"🦄🦄🔴🔴Error: Web directory not found: {dist_path}")


# ComfyUI imports this file as a package. Imported any other way (e.g. by pytest
# while collecting the repo) there is no ComfyUI to register with.
if __package__:
    _register_with_comfyui()
