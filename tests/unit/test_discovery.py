"""Node, model and workflow discovery against a fake ComfyUI install."""

import asyncio
import json
import os

import pytest

from backend.discovery import models, nodes, workflows
from backend.discovery.search import query_tokens, rank, tokenize
from backend.utils.workflow_validation import validate_api_workflow
from tests.fixtures.comfy_env import OBJECT_INFO, PONY_UI, TXT2IMG_API, install, write_workflow


@pytest.fixture
def comfy(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def test_tokenize_splits_camel_case_and_separators():
    assert tokenize("KSamplerAdvanced") == ["ksampler", "advanced"]
    assert tokenize("pony/ponyDiffusionV6XL.safetensors") == ["pony", "pony", "diffusion", "v6", "xl", "safetensors"]
    assert query_tokens(["upscale", "image"], "the 4x") == ["upscale", "4x"]


def test_rank_prefers_items_matching_more_of_the_query():
    items = ["flux_realism_lora", "add_detail_lora", "flux_dev"]
    ranked = [item for _, item in rank(items, ["flux", "lora"], lambda item: [(item, 1)])]
    assert ranked[0] == "flux_realism_lora"
    assert set(ranked) == set(items)


# --- nodes ---

def test_installed_custom_nodes_are_discoverable(comfy):
    definitions = asyncio.run(nodes.get_node_definitions())
    hits = nodes.search_nodes(definitions, ["upscale"])
    assert hits[0]["class_name"] == "UltimateSDUpscale"
    assert hits[0]["pack"] == "ComfyUI_UltimateSDUpscale"
    assert nodes.search_nodes(definitions, "lora")[0]["class_name"] == "LoraLoader"
    assert nodes.search_nodes(definitions, "") == []


def test_find_class_is_forgiving_about_case_and_display_names():
    assert nodes.find_class(OBJECT_INFO, "ksampler") == "KSampler"
    assert nodes.find_class(OBJECT_INFO, "Load Checkpoint") == "CheckpointLoaderSimple"
    assert nodes.find_class(OBJECT_INFO, "FaceDetailer") is None


def test_summarize_node_reads_both_combo_formats_and_types_outputs():
    summary = nodes.summarize_node("KSampler", OBJECT_INFO["KSampler"])
    assert summary["inputs"]["sampler_name"] == {"type": "COMBO", "options": ["euler", "dpmpp_2m"], "required": True}
    assert summary["inputs"]["scheduler"]["options"] == ["normal", "karras"]
    assert summary["inputs"]["cfg"]["min"] == 0.0
    loader = nodes.summarize_node("CheckpointLoaderSimple", OBJECT_INFO["CheckpointLoaderSimple"], max_options=1)
    assert loader["inputs"]["ckpt_name"]["options_total"] == 2
    assert loader["outputs"][2] == {"index": 2, "name": "VAE", "type": "VAE"}


def test_node_definitions_are_cached_and_survive_comfyui_going_away(comfy, monkeypatch):
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) > 1:
            raise ConnectionError("ComfyUI restarted")
        return OBJECT_INFO

    monkeypatch.setattr(nodes, "get_object_info", flaky)
    nodes.clear_node_cache()
    assert asyncio.run(nodes.get_node_definitions()) is OBJECT_INFO
    assert asyncio.run(nodes.get_node_definitions()) is OBJECT_INFO
    assert len(calls) == 1
    # A failed refresh keeps the last good list
    assert asyncio.run(nodes.get_node_definitions(refresh=True)) is OBJECT_INFO
    assert len(calls) == 2


def test_missing_nodes_are_reported():
    assert nodes.missing_node_types({"KSampler", "FaceDetailer"}, OBJECT_INFO) == ["FaceDetailer"]
    assert nodes.missing_node_types({"FaceDetailer"}, None) == []


# --- models ---

def test_models_listed_per_folder_without_non_model_folders(comfy):
    inventory = models.list_models()
    assert "custom_nodes" not in inventory
    assert inventory["checkpoints"] == sorted(["sd_xl_base_1.0.safetensors", "pony/ponyDiffusionV6XL.safetensors"])
    assert models.list_models("nope") == {}


def test_search_models_by_family_and_folder(comfy):
    assert models.search_models("pony")[0] == {"folder": "checkpoints", "name": "pony/ponyDiffusionV6XL.safetensors"}
    assert [hit["name"] for hit in models.search_models("flux lora")][0] == "flux/flux_realism_lora.safetensors"
    assert {hit["folder"] for hit in models.search_models("", folder="vae")} == {"vae"}


def test_installed_names_match_either_slash_and_bare_names(comfy):
    names = models.installed_names()
    assert "pony/ponydiffusionv6xl.safetensors" in names
    assert "ponydiffusionv6xl.safetensors" in names


# --- workflows ---

def test_ui_workflow_description_skips_virtual_nodes_and_finds_models():
    described = workflows.describe_workflow(PONY_UI)
    assert described["format"] == "ui"
    # Notes, reroutes and the subgraph instance aren't node types; the subgraph's contents are
    assert described["node_types"] == ["CheckpointLoaderSimple", "FaceDetailer", "KSampler", "LoraLoader",
                                       "UltimateSDUpscale", "VAEDecode"]
    assert described["models"] == ["pony/ponyDiffusionV6XL.safetensors", "missing_style_lora.safetensors",
                                   "sdxl_vae.safetensors"]
    assert described["notes"] == "Pony portrait workflow with a style LoRA."


def test_api_workflow_description_and_non_workflows():
    described = workflows.describe_workflow(TXT2IMG_API)
    assert described["format"] == "api"
    assert "KSampler" in described["node_types"]
    assert described["models"] == ["sd_xl_base_1.0.safetensors"]
    assert workflows.describe_workflow({"name": "package.json"}) is None
    assert workflows.describe_workflow([1, 2]) is None


def test_index_covers_comfyui_user_workflows_and_configured_dirs(comfy):
    user_dir, extra = comfy
    write_workflow(user_dir, "SDXL txt2img.json", TXT2IMG_API)
    write_workflow(extra, "portraits/pony portrait.json", PONY_UI)
    write_workflow(extra, "settings.json", {"theme": "dark"})
    with open(os.path.join(extra, "broken.json"), "w") as f:
        f.write("{not json")

    entries = {entry.name: entry for entry in workflows.workflow_index.entries()}
    assert set(entries) == {"SDXL txt2img", "pony portrait"}
    assert entries["SDXL txt2img"].label == "ComfyUI/SDXL txt2img.json"
    assert entries["pony portrait"].label == "my-workflows/portraits/pony portrait.json"
    assert "path" not in entries["pony portrait"].public()

    assert [entry.name for entry in workflows.search_workflows("pony lora")] == ["pony portrait"]
    assert [entry.name for entry in workflows.search_workflows("sdxl")][0] == "SDXL txt2img"


def test_index_picks_up_changes_and_deletions(comfy):
    user_dir, _ = comfy
    path = write_workflow(user_dir, "a.json", TXT2IMG_API)
    assert [entry.name for entry in workflows.workflow_index.entries()] == ["a"]
    write_workflow(user_dir, "a.json", PONY_UI)
    os.utime(path, (1, 1))  # different mtime even on coarse filesystem clocks
    assert workflows.workflow_index.entries()[0].format == "ui"
    os.remove(path)
    assert workflows.workflow_index.entries() == []


def test_workflow_ids_are_stable_and_lookup_works(comfy):
    user_dir, _ = comfy
    write_workflow(user_dir, "a.json", TXT2IMG_API)
    first = workflows.workflow_index.entries()[0]
    assert workflows.WorkflowIndex().get(first.id).name == "a"
    assert workflows.workflow_index.get("0" * 16) is None


def test_workflow_requirements_report_missing_nodes_and_models(comfy):
    _, extra = comfy
    write_workflow(extra, "pony.json", PONY_UI)
    entry = workflows.workflow_index.entries()[0]
    missing = workflows.workflow_requirements(entry, OBJECT_INFO, models.installed_names())
    assert missing == {"missing_nodes": ["FaceDetailer"], "missing_models": ["missing_style_lora.safetensors"]}
    # Without ComfyUI's lists nothing is claimed to be missing
    assert workflows.workflow_requirements(entry, None, None) == {"missing_nodes": [], "missing_models": []}


def test_thumbnail_next_to_workflow_is_found(comfy):
    user_dir, _ = comfy
    write_workflow(user_dir, "a.json", TXT2IMG_API)
    (user_dir / "a.webp").write_bytes(b"RIFF")
    entry = workflows.workflow_index.entries()[0]
    assert entry.thumbnail.endswith("a.webp")
    assert entry.public()["has_thumbnail"] is True


# --- validation against discovered definitions ---

def _broken(node_id, **inputs):
    workflow = json.loads(json.dumps(TXT2IMG_API))
    workflow[node_id]["inputs"].update(inputs)
    return workflow


def test_real_txt2img_workflow_is_valid():
    assert validate_api_workflow(TXT2IMG_API, OBJECT_INFO) == []


@pytest.mark.parametrize("workflow,expected", [
    # CLIP output wired into a MODEL input
    (_broken("3", model=["4", 1]),
     "Node 3 (KSampler) input 'model' expects MODEL but is linked to a CLIP output of node 4 (CheckpointLoaderSimple)."),
    (_broken("8", vae=["4", 3]),
     "Node 8 (VAEDecode) input 'vae' links to output 3 of node 4 (CheckpointLoaderSimple), which has 3 outputs."),
    (_broken("4", ckpt_name="dreamshaper_8.safetensors"),
     "Node 4 (CheckpointLoaderSimple) input 'ckpt_name' is 'dreamshaper_8.safetensors', which is not an available "
     "option (sd_xl_base_1.0.safetensors, pony/ponyDiffusionV6XL.safetensors)."),
    (_broken("3", sampler_name="euler_ancestral_magic"),
     "Node 3 (KSampler) input 'sampler_name' is 'euler_ancestral_magic', which is not an available option "
     "(euler, dpmpp_2m)."),
    (_broken("3", steps="many"), "Node 3 (KSampler) input 'steps' must be a number (INT), got 'many'."),
    (_broken("3", denoise=1.5), "Node 3 (KSampler) input 'denoise' is 1.5, above the maximum 1.0."),
    (_broken("5", width=8), "Node 5 (EmptyLatentImage) input 'width' is 8, below the minimum 16."),
])
def test_invalid_input_types_and_values_are_rejected(workflow, expected):
    assert validate_api_workflow(workflow, OBJECT_INFO) == [expected]


def test_numeric_strings_and_wildcards_pass_like_in_comfyui():
    workflow = _broken("3", steps="30")
    workflow["10"] = {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}}
    assert validate_api_workflow(workflow, OBJECT_INFO) == []


def test_comma_separated_types_match_when_they_overlap():
    definitions = {**OBJECT_INFO, "Masker": {"input": {"required": {"pixels": ["IMAGE,MASK"]}}, "output": []}}
    workflow = {**TXT2IMG_API, "10": {"class_type": "Masker", "inputs": {"pixels": ["8", 0]}}}
    assert validate_api_workflow(workflow, definitions) == []
    workflow["10"]["inputs"]["pixels"] = ["5", 0]
    assert validate_api_workflow(workflow, definitions) == [
        "Node 10 (Masker) input 'pixels' expects IMAGE,MASK but is linked to a LATENT output of node 5 (EmptyLatentImage)."]
