"""The local workflow library as the chat sees it: search results plus what each one is missing."""

import asyncio
from typing import Any, Optional

from ..utils.logger import log
from .models import installed_names
from .nodes import get_node_definitions
from .workflows import WorkflowEntry, search_workflows, workflow_requirements

# Marks cards the UI loads from /api/discovery/workflows/{id} instead of the hosted server
LOCAL_WORKFLOW_SOURCE = "Local library"


def workflow_card(entry: WorkflowEntry, missing: dict[str, list[str]]) -> dict[str, Any]:
    """The shape the chat UI renders as a workflow option (see WorkflowOption.tsx)."""
    description = entry.notes or ("Nodes: " + ", ".join(entry.node_types[:12]) if entry.node_types else "")
    if missing["missing_nodes"]:
        description += f" Missing nodes: {', '.join(missing['missing_nodes'])}."
    if missing["missing_models"]:
        description += f" Missing models: {', '.join(missing['missing_models'])}."
    return {
        "id": entry.id,
        "name": entry.name,
        "description": description.strip(),
        "image": f"/api/discovery/workflows/{entry.id}/thumbnail" if entry.thumbnail else "",
        "source": LOCAL_WORKFLOW_SOURCE,
    }


async def recall_local_workflows(query: str, limit: int = 6) -> dict[str, Any]:
    entries = await asyncio.to_thread(search_workflows, query, limit)
    definitions = await get_node_definitions() if entries else None
    installed: Optional[set[str]] = None
    if entries:
        try:
            installed = await asyncio.to_thread(installed_names)
        except Exception as e:
            log.warning(f"Could not list model files for workflow recall: {e}")

    cards, details = [], []
    for entry in entries:
        missing = workflow_requirements(entry, definitions, installed)
        cards.append(workflow_card(entry, missing))
        details.append({"id": entry.id, "name": entry.name, "file": entry.label, "format": entry.format,
                        "node_types": entry.node_types, "models": entry.models, **missing})
    if cards:
        answer = (f"Found {len(cards)} workflow(s) in the local library. They are shown to the user "
                  "with an Accept button that loads one onto the canvas.")
    else:
        answer = "No workflow in the local library matched. Offer to build one on the canvas instead."
    return {"answer": answer, "data": cards, "workflows": details}
