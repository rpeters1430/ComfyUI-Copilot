"""The model files installed in this ComfyUI: checkpoints, LoRAs, VAEs, text encoders, ...

Reads ComfyUI's own folder registry (`folder_paths`), so extra_model_paths.yaml entries and
folders added by custom nodes are included, with the same relative names ComfyUI's loader
nodes accept ("sdxl/juggernaut.safetensors").
"""

import os
from typing import Any, Optional

import folder_paths

from .search import query_tokens, rank

# Registered folders that don't hold models
NON_MODEL_FOLDERS = {"custom_nodes", "configs"}

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".gguf", ".sft", ".onnx")


def model_folders() -> list[str]:
    return sorted(name for name in folder_paths.folder_names_and_paths if name not in NON_MODEL_FOLDERS)


def list_models(folder: Optional[str] = None) -> dict[str, list[str]]:
    """{folder: [relative file names]}, for one folder or all of them. Unknown folders are empty."""
    folders = [folder] if folder else model_folders()
    inventory = {}
    for name in folders:
        if name not in folder_paths.folder_names_and_paths:
            continue
        try:
            inventory[name] = sorted(folder_paths.get_filename_list(name))
        except Exception:
            # A broken extra path shouldn't hide the other folders
            inventory[name] = []
    return inventory


def normalize_name(name: str) -> str:
    """Compare model names the way users write them: case-insensitive, either slash."""
    return name.replace("\\", "/").lower()


def installed_names(inventory: Optional[dict[str, list[str]]] = None) -> set[str]:
    """Every installed model by relative name and by bare file name, normalized."""
    names = set()
    for files in (inventory if inventory is not None else list_models()).values():
        for file in files:
            normalized = normalize_name(file)
            names.add(normalized)
            names.add(normalized.rsplit("/", 1)[-1])
    return names


def looks_like_model_file(value: Any) -> bool:
    return isinstance(value, str) and value.lower().endswith(MODEL_EXTENSIONS)


def search_models(query: Any = "", folder: Optional[str] = None, limit: int = 20,
                  inventory: Optional[dict[str, list[str]]] = None) -> list[dict[str, str]]:
    """Installed model files matching `query`, best first. An empty query lists them all."""
    inventory = inventory if inventory is not None else list_models(folder)
    entries = [(name, file) for name, files in inventory.items() for file in files]
    tokens = query_tokens(query)
    ranked = rank(entries, tokens, lambda entry: [(os.path.splitext(entry[1])[0], 10), (entry[0], 4)], limit)
    return [{"folder": name, "name": file} for _, (name, file) in ranked]
