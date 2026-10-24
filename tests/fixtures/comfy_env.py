"""A small fake ComfyUI install: node definitions, model folders and a workflow library."""

import json
import os

MODELS = {
    "checkpoints": ["sd_xl_base_1.0.safetensors", "pony/ponyDiffusionV6XL.safetensors"],
    "loras": ["add_detail.safetensors", "flux/flux_realism_lora.safetensors"],
    "vae": ["sdxl_vae.safetensors"],
    "upscale_models": ["4x-UltraSharp.pth"],
}

OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [MODELS["checkpoints"], {}]}},
        "output": ["MODEL", "CLIP", "VAE"], "output_name": ["MODEL", "CLIP", "VAE"],
        "display_name": "Load Checkpoint", "category": "loaders", "python_module": "nodes",
        "description": "Loads a diffusion model checkpoint.",
    },
    "LoraLoader": {
        "input": {"required": {"model": ["MODEL"], "clip": ["CLIP"], "lora_name": [MODELS["loras"]],
                               "strength_model": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}],
                               "strength_clip": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}]}},
        "output": ["MODEL", "CLIP"], "output_name": ["MODEL", "CLIP"],
        "display_name": "Load LoRA", "category": "loaders", "python_module": "nodes",
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
        "output": ["CONDITIONING"], "output_name": ["CONDITIONING"],
        "display_name": "CLIP Text Encode (Prompt)", "category": "conditioning", "python_module": "nodes",
    },
    "EmptyLatentImage": {
        "input": {"required": {"width": ["INT", {"default": 512, "min": 16, "max": 16384}],
                               "height": ["INT", {"default": 512, "min": 16, "max": 16384}],
                               "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}]}},
        "output": ["LATENT"], "output_name": ["LATENT"],
        "display_name": "Empty Latent Image", "category": "latent", "python_module": "nodes",
    },
    "KSampler": {
        "input": {"required": {"model": ["MODEL"], "seed": ["INT", {"min": 0, "max": 2**64 - 1}],
                               "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                               "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                               # Current ComfyUI combo format
                               "sampler_name": ["COMBO", {"options": ["euler", "dpmpp_2m"]}],
                               "scheduler": [["normal", "karras"]],
                               "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                               "latent_image": ["LATENT"],
                               "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}]}},
        "output": ["LATENT"], "output_name": ["LATENT"],
        "display_name": "KSampler", "category": "sampling", "python_module": "nodes",
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"], "output_name": ["IMAGE"],
        "display_name": "VAE Decode", "category": "latent", "python_module": "nodes",
    },
    "SaveImage": {
        "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]},
                  "hidden": {"prompt": "PROMPT"}},
        "output": [], "output_name": [],
        "display_name": "Save Image", "category": "image", "python_module": "nodes",
    },
    "PreviewAny": {
        "input": {"required": {"source": ["*"]}},
        "output": [], "display_name": "Preview Any", "category": "utils", "python_module": "comfy_extras.nodes_preview_any",
    },
    "UltimateSDUpscale": {
        "input": {"required": {"image": ["IMAGE"], "upscale_model": ["UPSCALE_MODEL"],
                               "upscale_by": ["FLOAT", {"default": 2.0, "min": 0.05, "max": 4.0}]}},
        "output": ["IMAGE"], "output_name": ["IMAGE"],
        "display_name": "Ultimate SD Upscale", "category": "image/upscaling",
        "python_module": "custom_nodes.ComfyUI_UltimateSDUpscale",
        "description": "Tiled upscaling with a diffusion pass per tile.",
    },
}

TXT2IMG_API = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "seed": 1, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
        "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0], "denoise": 1.0}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
}

# What the ComfyUI frontend saves: nodes with widgets_values, notes, reroutes, a subgraph,
# and template-style model metadata
PONY_UI = {
    "last_node_id": 12,
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["pony/ponyDiffusionV6XL.safetensors"]},
        {"id": 2, "type": "LoraLoader", "widgets_values": ["missing_style_lora.safetensors", 0.8, 0.8]},
        {"id": 3, "type": "KSampler", "widgets_values": [1, "randomize", 25, 6.5, "euler", "normal", 1.0]},
        {"id": 4, "type": "Note", "widgets_values": ["Pony portrait workflow with a style LoRA."]},
        {"id": 5, "type": "Reroute"},
        {"id": 6, "type": "FaceDetailer", "widgets_values": []},
        {"id": 7, "type": "1d2c3b4a-subgraph", "widgets_values": []},
        {"id": 8, "type": "VAEDecode", "properties": {"models": [
            {"name": "sdxl_vae.safetensors", "url": "https://example.invalid/sdxl_vae.safetensors",
             "directory": "vae"}]}},
    ],
    "links": [],
    "definitions": {"subgraphs": [{"id": "1d2c3b4a-subgraph", "nodes": [
        {"id": 1, "type": "UltimateSDUpscale", "widgets_values": [2.0]},
    ]}]},
}


def install(monkeypatch, tmp_path, object_info=OBJECT_INFO, models=MODELS):
    """Point the discovery layer at fake definitions, models and workflow folders.

    Returns (user workflows dir, extra workflows dir); both start empty.
    """
    import folder_paths

    from backend.discovery import nodes, workflows

    async def fake_object_info():
        return object_info

    monkeypatch.setattr(nodes, "get_object_info", fake_object_info)
    nodes.clear_node_cache()

    folders = {name: ([str(tmp_path / "models" / name)], {".safetensors"}) for name in models}
    folders["custom_nodes"] = ([str(tmp_path / "custom_nodes")], set())
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", folders, raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda name: list(models.get(name, [])), raising=False)

    user_dir = tmp_path / "user"
    user_workflows = user_dir / "default" / "workflows"
    user_workflows.mkdir(parents=True)
    extra = tmp_path / "my-workflows"
    extra.mkdir()
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir), raising=False)
    monkeypatch.setattr(workflows, "WORKFLOW_DIRS", [str(extra)])
    monkeypatch.setattr(workflows, "workflow_index", workflows.WorkflowIndex())
    return user_workflows, extra


def write_workflow(directory, name, workflow) -> str:
    path = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f)
    return path
