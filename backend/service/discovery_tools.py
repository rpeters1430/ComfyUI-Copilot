"""Agent tools over the discovery layer: installed model files and the local workflow library."""

import asyncio
import json
from typing import Any

from agents.tool import function_tool

from ..discovery.library import recall_local_workflows
from ..discovery.models import list_models, search_models as find_models
from ..utils.logger import log


@function_tool
async def search_models(query: str = "", folder: str = "", limit: int = 20) -> str:
    """
    Find model files installed in this ComfyUI. Use the returned `name` exactly as the value
    of a loader input (ckpt_name, lora_name, vae_name, unet_name, clip_name, ...); never invent
    file names.
    - query: words from the model name or family ("sdxl", "flux dev", "4x upscale"). Empty lists all.
    - folder: optional ComfyUI model folder ("checkpoints", "loras", "vae", "diffusion_models",
      "text_encoders", "clip_vision", "controlnet", "upscale_models", ...).
    """
    try:
        inventory = await asyncio.to_thread(list_models, folder or None)
    except Exception as e:
        return json.dumps({"error": f"Could not list model files: {e}"})
    if folder and folder not in inventory:
        return json.dumps({"error": f"Unknown model folder '{folder}'",
                           "folders": sorted(await asyncio.to_thread(list_models))})
    limit = max(1, min(int(limit or 20), 100))
    results = find_models(query, inventory=inventory, limit=limit)
    response: dict[str, Any] = {"query": query, "folder": folder, "results": results}
    if not results:
        response["message"] = "No installed model files matched"
        response["folders"] = {name: len(files) for name, files in inventory.items() if files}
    return json.dumps(response, ensure_ascii=False)


@function_tool
async def recall_workflow(query: str, limit: int = 6) -> str:
    """
    Search the user's local workflow library (workflows saved in ComfyUI and configured folders)
    for workflows matching what they want to make ("sdxl lora portrait", "flux inpaint",
    "wan image to video"). Matching workflows are shown to the user as cards they can load.
    Each result lists the node types and model files it needs that aren't installed.
    """
    limit = max(1, min(int(limit or 6), 12))
    try:
        result = await recall_local_workflows(query, limit)
    except Exception as e:
        log.error(f"Local workflow recall failed: {e}")
        return json.dumps({"error": f"Could not search the local workflow library: {e}"})
    return json.dumps(result, ensure_ascii=False)
