// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

// Calls to the local plugin backend's provider routes (backend/controller/provider_api.py)

export type ProviderKind = "openai" | "ollama" | "lmstudio";

export interface ModelCapabilities {
  tools: boolean | null;
  vision: boolean | null;
  structured_output: boolean | null;
  context_length: number | null;
}

export interface ProviderModel {
  id: string;
  provider: ProviderKind;
  type: "llm" | "vlm" | "embedding";
  display_name: string | null;
  loaded: boolean | null;
  capabilities: ModelCapabilities;
}

export interface ProviderStatus {
  ok: boolean;
  kind: ProviderKind;
  base_url: string;
  message: string;
  version: string | null;
  models: ProviderModel[];
  warnings: string[];
}

export async function testProvider(kind: ProviderKind, baseUrl: string, apiKey: string): Promise<ProviderStatus> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey.trim()) headers["Openai-Api-Key"] = apiKey.trim();
  const response = await fetch("/api/providers/test", {
    method: "POST",
    headers,
    body: JSON.stringify({ kind, base_url: baseUrl }),
  });
  const result = await response.json();
  if (!result.data) {
    throw new Error(result.message || `Connection test failed (HTTP ${response.status})`);
  }
  return result.data as ProviderStatus;
}

export async function discoverProviders(): Promise<ProviderStatus[]> {
  const response = await fetch("/api/providers/discover");
  const result = await response.json();
  return (result.data || []) as ProviderStatus[];
}
