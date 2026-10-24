// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

import type { ProviderKind } from "../../apis/providerApi";

export type ToolMode = "auto" | "agent" | "json";

export interface ProviderSettingsValue {
  provider: ProviderKind;
  baseUrl: string;
  apiKey: string;
  workflowModel: string;
  visionModel: string;
  toolMode: ToolMode;
}

export const PROVIDER_OPTIONS: { kind: ProviderKind; label: string; defaultBaseUrl: string; needsApiKey: boolean }[] = [
  { kind: "openai", label: "OpenAI-compatible", defaultBaseUrl: "https://api.openai.com/v1", needsApiKey: true },
  { kind: "ollama", label: "Ollama", defaultBaseUrl: "http://localhost:11434/v1", needsApiKey: false },
  { kind: "lmstudio", label: "LM Studio", defaultBaseUrl: "http://localhost:1234/v1", needsApiKey: false },
];

export const providerOption = (kind: ProviderKind) =>
  PROVIDER_OPTIONS.find((option) => option.kind === kind) ?? PROVIDER_OPTIONS[0];

// Keep in sync with backend/providers/registry.py detect_kind
export function detectProviderKind(baseUrl: string): ProviderKind {
  let port = "";
  try {
    port = new URL(baseUrl).port;
  } catch {
    return "openai";
  }
  if (port === "11434") return "ollama";
  if (port === "1234" || port === "1235") return "lmstudio";
  return "openai";
}

const KEYS = {
  provider: "llmProvider",
  baseUrl: "openaiBaseUrl",
  apiKey: "openaiApiKey",
  workflowModel: "workflowLLMModel",
  visionModel: "visionLLMModel",
  toolMode: "copilotToolMode",
} as const;

/** Read saved settings. Installs from before providers existed only have a base URL; infer the provider. */
export function loadProviderSettings(): ProviderSettingsValue {
  const baseUrl = localStorage.getItem(KEYS.baseUrl) || "";
  const savedProvider = localStorage.getItem(KEYS.provider) as ProviderKind | null;
  const toolMode = localStorage.getItem(KEYS.toolMode) as ToolMode | null;
  return {
    provider: savedProvider && PROVIDER_OPTIONS.some((o) => o.kind === savedProvider)
      ? savedProvider
      : detectProviderKind(baseUrl),
    baseUrl,
    apiKey: localStorage.getItem(KEYS.apiKey) || "",
    workflowModel: localStorage.getItem(KEYS.workflowModel) || "",
    visionModel: localStorage.getItem(KEYS.visionModel) || "",
    toolMode: toolMode === "agent" || toolMode === "json" ? toolMode : "auto",
  };
}

/** Save settings; returns true if anything changed. */
export function saveProviderSettings(value: ProviderSettingsValue): boolean {
  let changed = false;
  for (const field of Object.keys(KEYS) as (keyof ProviderSettingsValue)[]) {
    const key = KEYS[field];
    const next = String(value[field] ?? "").trim();
    const stored = field === "toolMode" && next === "auto" ? "" : next;
    if ((localStorage.getItem(key) || "") !== stored) changed = true;
    if (stored) localStorage.setItem(key, stored);
    else localStorage.removeItem(key);
  }
  return changed;
}
