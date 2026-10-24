// Copyright (C) 2025 AIDC-AI
// Licensed under the MIT License.

import { useEffect, useState } from "react";
import { discoverProviders, ProviderModel, ProviderStatus, testProvider } from "../../apis/providerApi";
import { PROVIDER_OPTIONS, ProviderSettingsValue, ToolMode, providerOption } from "./providerStorage";

interface ProviderSettingsProps {
  value: ProviderSettingsValue;
  onChange: (patch: Partial<ProviderSettingsValue>) => void;
  // The workflow model is set in the separate-endpoint card when that endpoint is used
  separateWorkflowEndpoint: boolean;
}

const inputClass = `w-full px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-lg text-xs
  bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400
  focus:border-blue-500 dark:focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20
  focus:outline-none`;
const labelClass = "block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2";
const buttonClass = (busy: boolean) => `px-4 py-2 rounded-lg font-medium text-xs transition-colors ${
  busy
    ? "bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed"
    : "bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white"
}`;

const TOOL_MODES: { value: ToolMode; label: string }[] = [
  { value: "auto", label: "Auto: full assistant if the model supports tool calling" },
  { value: "agent", label: "Always use tool calling" },
  { value: "json", label: "Simple mode: no tool calling (for small local models)" },
];

export function describeCapabilities(model: ProviderModel): string {
  const { tools, vision, context_length } = model.capabilities;
  const parts = [];
  if (tools === true) parts.push("tools");
  if (tools === false) parts.push("no tool calling");
  if (vision) parts.push("vision");
  if (context_length) parts.push(`${Math.round(context_length / 1024)}k context`);
  return parts.join(" · ");
}

function countModels(status: ProviderStatus): string {
  const count = status.models.filter((model) => model.type !== "embedding").length;
  return `${count} ${count === 1 ? "model" : "models"}`;
}

function ModelSelect({ label, value, onChange, models, emptyLabel, hint }: {
  label: string; value: string; onChange: (value: string) => void; models: ProviderModel[];
  emptyLabel: string; hint: string;
}) {
  // Keep a saved model selectable even when the server doesn't list it right now
  const known = models.some((model) => model.id === value);
  return (
    <div className="mb-4">
      <label className={labelClass}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className={inputClass}>
        <option value="">{emptyLabel}</option>
        {!known && value && <option value={value}>{value}</option>}
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.display_name || model.id}{describeCapabilities(model) ? ` (${describeCapabilities(model)})` : ""}
          </option>
        ))}
      </select>
      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">{hint}</div>
    </div>
  );
}

export function ProviderSettings({ value, onChange, separateWorkflowEndpoint }: ProviderSettingsProps) {
  const [status, setStatus] = useState<ProviderStatus | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [discovered, setDiscovered] = useState<ProviderStatus[] | null>(null);
  const [detecting, setDetecting] = useState(false);
  const option = providerOption(value.provider);
  const chatModels = (status?.models || []).filter((model) => model.type !== "embedding");
  const visionModels = chatModels.filter((model) => model.capabilities.vision !== false);

  const runTest = async (kind = value.provider, baseUrl = value.baseUrl || option.defaultBaseUrl) => {
    setTesting(true);
    setTestError(null);
    try {
      setStatus(await testProvider(kind, baseUrl, value.apiKey));
    } catch (error) {
      setStatus(null);
      setTestError(error instanceof Error ? error.message : "Connection test failed");
    } finally {
      setTesting(false);
    }
  };

  // Load the model list for the saved provider when the settings open
  useEffect(() => {
    if (value.baseUrl) runTest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDetect = async () => {
    setDetecting(true);
    try {
      setDiscovered(await discoverProviders());
    } finally {
      setDetecting(false);
    }
  };

  const applyDiscovered = (found: ProviderStatus) => {
    onChange({ provider: found.kind, baseUrl: found.base_url, apiKey: "" });
    setStatus(found);
    setTestError(null);
    setDiscovered(null);
  };

  return (
    <div>
      <div className="mb-4">
        <label className={labelClass}>Provider</label>
        <div className="flex gap-2">
          <select
            value={value.provider}
            onChange={(e) => {
              const next = providerOption(e.target.value as ProviderSettingsValue["provider"]);
              // Switching provider resets a URL that was just the previous provider's default
              const isDefaultUrl = !value.baseUrl || PROVIDER_OPTIONS.some((o) => o.defaultBaseUrl === value.baseUrl);
              onChange({ provider: next.kind, ...(isDefaultUrl ? { baseUrl: next.defaultBaseUrl } : {}) });
              setStatus(null);
            }}
            className={inputClass}
          >
            {PROVIDER_OPTIONS.map((o) => <option key={o.kind} value={o.kind}>{o.label}</option>)}
          </select>
          <button type="button" onClick={handleDetect} disabled={detecting}
                  className={`${buttonClass(detecting)} whitespace-nowrap`}>
            {detecting ? "Detecting..." : "Detect local"}
          </button>
        </div>
        {discovered && (
          <div className="mt-2 text-xs">
            {discovered.length === 0 && (
              <div className="text-gray-500 dark:text-gray-400">
                No Ollama or LM Studio server found on this computer's default ports.
              </div>
            )}
            {discovered.map((found) => (
              <button key={found.base_url} type="button" onClick={() => applyDiscovered(found)}
                      className="block w-full text-left mb-1 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-600
                                 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-200">
                Use {providerOption(found.kind).label} at {found.base_url} ({countModels(found)})
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="mb-4">
        <label className={labelClass}>Server URL</label>
        <input type="text" value={value.baseUrl} placeholder={option.defaultBaseUrl}
               onChange={(e) => onChange({ baseUrl: e.target.value })} className={inputClass} />
      </div>

      <div className="mb-4">
        <label className={labelClass}>API Key{option.needsApiKey ? "" : " (usually not needed)"}</label>
        <input type="password" value={value.apiKey} onChange={(e) => onChange({ apiKey: e.target.value })}
               placeholder={option.needsApiKey ? "Enter your API key" : "Leave empty for a local server"}
               className={inputClass} />
      </div>

      <div className="flex items-center mb-2">
        <button type="button" onClick={() => runTest()} disabled={testing} className={buttonClass(testing)}>
          {testing ? "Testing..." : "Test connection"}
        </button>
      </div>
      {(status || testError) && (
        <div className={`text-xs p-2 rounded-md mb-4 ${status?.ok
          ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400"
          : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400"}`}>
          {testError || status?.message}
          {status?.warnings.map((warning) => (
            <div key={warning} className="mt-1 text-amber-700 dark:text-amber-400">{warning}</div>
          ))}
        </div>
      )}

      {!separateWorkflowEndpoint && (
        <ModelSelect label="Workflow model (debug and canvas edits)" value={value.workflowModel}
                     onChange={(workflowModel) => onChange({ workflowModel })} models={chatModels}
                     emptyLabel="Same as the chat model"
                     hint="Editing workflows needs a strong model with a large context window." />
      )}
      <ModelSelect label="Vision model" value={value.visionModel}
                   onChange={(visionModel) => onChange({ visionModel })} models={visionModels}
                   emptyLabel="None"
                   hint="Describes attached images when the chat model can't see them." />

      <div className="mb-2">
        <label className={labelClass}>Tool calling</label>
        <select value={value.toolMode} onChange={(e) => onChange({ toolMode: e.target.value as ToolMode })}
                className={inputClass}>
          {TOOL_MODES.map((mode) => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
        </select>
      </div>
    </div>
  );
}
