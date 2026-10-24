"""A searchable index of workflow JSON files on this computer.

Indexed folders: the workflows saved in ComfyUI's user directory (the frontend's Workflows
panel) plus any folders listed in COPILOT_WORKFLOW_DIRS. Nothing else on disk is read.
Both UI-format (what the frontend saves) and API-format workflows are understood.
Files are re-parsed only when their size or modification time changes.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import folder_paths

from ..utils.globals import WORKFLOW_DIRS
from ..utils.logger import log
from .models import looks_like_model_file, normalize_name
from .search import query_tokens, rank

MAX_FILES = 5000
MAX_FILE_BYTES = 20 * 1024 * 1024
THUMBNAIL_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")

# Frontend-only node types: they never appear in /object_info, so they're not "missing"
VIRTUAL_NODE_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
NOTE_NODE_TYPES = {"Note", "MarkdownNote"}


@dataclass
class WorkflowEntry:
    id: str
    name: str
    path: str
    label: str
    format: str
    node_types: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    notes: str = ""
    thumbnail: Optional[str] = None

    def public(self) -> dict[str, Any]:
        """The entry without its absolute path, for the browser and the LLM."""
        data = asdict(self)
        data.pop("path")
        data["has_thumbnail"] = bool(data.pop("thumbnail"))
        return data


def workflow_dirs() -> list[tuple[str, str]]:
    """(label, absolute directory) pairs that exist, in priority order, without duplicates."""
    candidates = []
    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    if get_user_directory:
        try:
            candidates.append(("ComfyUI", os.path.join(get_user_directory(), "default", "workflows")))
        except Exception:
            pass
    for directory in WORKFLOW_DIRS:
        candidates.append((os.path.basename(os.path.normpath(directory)) or directory, directory))

    seen, result = set(), []
    for label, directory in candidates:
        real = os.path.realpath(os.path.expanduser(directory))
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            result.append((label, real))
    return result


def workflow_id(path: str) -> str:
    return hashlib.sha1(os.path.normcase(os.path.realpath(path)).encode("utf-8")).hexdigest()[:16]


def _ui_nodes(workflow: dict) -> tuple[list[dict], set[str]]:
    """All nodes of a UI-format workflow including subgraph contents, plus the subgraph ids."""
    nodes = [node for node in workflow.get("nodes") or [] if isinstance(node, dict)]
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs") or []
    subgraph_ids = set()
    for subgraph in subgraphs:
        if isinstance(subgraph, dict):
            subgraph_ids.add(str(subgraph.get("id")))
            nodes.extend(node for node in subgraph.get("nodes") or [] if isinstance(node, dict))
    return nodes, subgraph_ids


def describe_workflow(workflow: Any) -> Optional[dict[str, Any]]:
    """{format, node_types, models, notes} of a parsed workflow, or None if it isn't one."""
    if not isinstance(workflow, dict):
        return None
    node_types, models, notes = set(), [], []

    if isinstance(workflow.get("nodes"), list):
        nodes, subgraph_ids = _ui_nodes(workflow)
        for node in nodes:
            node_type = str(node.get("type") or "")
            if not node_type:
                continue
            values = node.get("widgets_values")
            values = values if isinstance(values, list) else []
            if node_type in NOTE_NODE_TYPES:
                notes.extend(value for value in values if isinstance(value, str))
            # Subgraph instances and legacy group nodes ("workflow>Name") aren't installable types
            if node_type not in VIRTUAL_NODE_TYPES and node_type not in subgraph_ids \
                    and not node_type.startswith(("workflow>", "workflow/")):
                node_types.add(node_type)
            models.extend(value for value in values if looks_like_model_file(value))
            # Templates list the models they need, with download links, in node properties
            for model in (node.get("properties") or {}).get("models") or []:
                if isinstance(model, dict) and looks_like_model_file(model.get("name")):
                    models.append(model["name"])
        workflow_format = "ui"
    else:
        api_nodes = [node for node in workflow.values() if isinstance(node, dict) and "class_type" in node]
        if not api_nodes or len(api_nodes) != len(workflow):
            return None
        for node in api_nodes:
            node_types.add(str(node["class_type"]))
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            models.extend(value for value in inputs.values() if looks_like_model_file(value))
        workflow_format = "api"

    return {
        "format": workflow_format,
        "node_types": sorted(node_types),
        "models": list(dict.fromkeys(models)),
        "notes": " ".join(" ".join(notes).split())[:500],
    }


def _thumbnail_for(path: str) -> Optional[str]:
    stem = os.path.splitext(path)[0]
    for candidate in [stem + ext for ext in THUMBNAIL_EXTENSIONS] + [stem + "-1.webp"]:
        if os.path.isfile(candidate):
            return candidate
    return None


class WorkflowIndex:
    def __init__(self):
        # path -> ((mtime, size), entry or None for files that aren't workflows)
        self._files: dict[str, tuple[tuple[float, int], Optional[WorkflowEntry]]] = {}

    def _scan_files(self) -> list[tuple[str, str, str]]:
        found = []
        for label, directory in workflow_dirs():
            for root, dirs, files in os.walk(directory):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for file in sorted(files):
                    if file.lower().endswith(".json"):
                        found.append((label, directory, os.path.join(root, file)))
                        if len(found) >= MAX_FILES:
                            log.warning(f"Workflow index stopped at {MAX_FILES} files")
                            return found
        return found

    def _load_entry(self, label: str, directory: str, path: str, stat: os.stat_result) -> Optional[WorkflowEntry]:
        if stat.st_size > MAX_FILE_BYTES:
            return None
        try:
            with open(path, encoding="utf-8") as f:
                description = describe_workflow(json.load(f))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if description is None:
            return None
        relative = os.path.relpath(path, directory).replace(os.sep, "/")
        return WorkflowEntry(
            id=workflow_id(path),
            name=os.path.splitext(os.path.basename(path))[0],
            path=path,
            label=f"{label}/{relative}",
            thumbnail=_thumbnail_for(path),
            **description,
        )

    def entries(self) -> list[WorkflowEntry]:
        """Current entries; changed files are re-read and deleted ones dropped."""
        current: dict[str, tuple[tuple[float, int], Optional[WorkflowEntry]]] = {}
        for label, directory, path in self._scan_files():
            try:
                stat = os.stat(path)
            except OSError:
                continue
            key = (stat.st_mtime, stat.st_size)
            cached = self._files.get(path)
            current[path] = cached if cached and cached[0] == key else \
                (key, self._load_entry(label, directory, path, stat))
        self._files = current
        return [entry for _, entry in current.values() if entry is not None]

    def get(self, entry_id: str) -> Optional[WorkflowEntry]:
        return next((entry for entry in self.entries() if entry.id == entry_id), None)


workflow_index = WorkflowIndex()


def _search_fields(entry: WorkflowEntry):
    return [
        (entry.name, 10),
        (entry.label, 4),
        (entry.notes, 3),
        (" ".join(entry.node_types), 3),
        (" ".join(entry.models), 3),
    ]


def search_workflows(query: Any = "", limit: int = 10, index: Optional[WorkflowIndex] = None) -> list[WorkflowEntry]:
    """Indexed workflows matching `query`, best first. An empty query lists them all."""
    entries = (index or workflow_index).entries()
    return [entry for _, entry in rank(entries, query_tokens(query), _search_fields, limit)]


def workflow_requirements(entry: WorkflowEntry, definitions: Optional[dict[str, Any]],
                          installed_models: Optional[set[str]]) -> dict[str, list[str]]:
    """Node types and model files the workflow uses that this ComfyUI doesn't have.

    Unknown (None) definitions or models mean "can't tell" and report nothing missing.
    """
    missing_nodes = [] if definitions is None else [t for t in entry.node_types if t not in definitions]
    missing_models = [] if installed_models is None else [
        model for model in entry.models
        if normalize_name(model) not in installed_models
        and normalize_name(model).rsplit("/", 1)[-1] not in installed_models
    ]
    return {"missing_nodes": missing_nodes, "missing_models": missing_models}


def load_workflow(entry: WorkflowEntry) -> Any:
    with open(entry.path, encoding="utf-8") as f:
        return json.load(f)
