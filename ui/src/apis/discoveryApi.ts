// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

// Calls to the local plugin backend's discovery routes (backend/controller/discovery_api.py)

// Matches LOCAL_WORKFLOW_SOURCE in backend/discovery/library.py
export const LOCAL_WORKFLOW_SOURCE = "Local library";

export interface LibraryWorkflow {
  id: string;
  name: string;
  label: string;
  format: "ui" | "api";
  node_types: string[];
  models: string[];
  workflow: any;
}

export async function getLibraryWorkflow(id: string): Promise<LibraryWorkflow> {
  const response = await fetch(`/api/discovery/workflows/${encodeURIComponent(id)}`);
  const result = await response.json().catch(() => ({}));
  if (!response.ok || !result.data) {
    throw new Error(result.message || `Could not load the workflow (HTTP ${response.status})`);
  }
  return result.data as LibraryWorkflow;
}
