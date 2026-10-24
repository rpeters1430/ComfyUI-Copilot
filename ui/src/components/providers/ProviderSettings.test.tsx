import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";

import type { ProviderStatus } from "../../apis/providerApi";
import { ProviderSettings } from "./ProviderSettings";
import type { ProviderSettingsValue } from "./providerStorage";

const lmStudio: ProviderStatus = {
  ok: true, kind: "lmstudio", base_url: "http://127.0.0.1:1235/v1",
  message: "Connected to LM Studio: 2 chat models available.", version: null, warnings: [],
  models: [
    { id: "qwen2.5-coder-7b", provider: "lmstudio", type: "llm", display_name: "Qwen2.5 Coder 7B", loaded: false,
      capabilities: { tools: false, vision: false, structured_output: true, context_length: 32768 } },
    { id: "qwen2.5-vl-7b", provider: "lmstudio", type: "vlm", display_name: null, loaded: false,
      capabilities: { tools: true, vision: true, structured_output: true, context_length: 32768 } },
    { id: "nomic-embed", provider: "lmstudio", type: "embedding", display_name: null, loaded: false,
      capabilities: { tools: null, vision: false, structured_output: true, context_length: 2048 } },
  ],
};

const empty: ProviderSettingsValue = {
  provider: "openai", baseUrl: "", apiKey: "", workflowModel: "", visionModel: "", toolMode: "auto",
};

let latest: ProviderSettingsValue = empty;

function Harness({ initial = empty }: { initial?: ProviderSettingsValue }) {
  const [value, setValue] = useState(initial);
  latest = value;
  return <ProviderSettings value={value} separateWorkflowEndpoint={false}
                           onChange={(patch) => setValue((prev) => ({ ...prev, ...patch }))} />;
}

function mockFetch(routes: Record<string, unknown>) {
  const fetchMock = vi.fn(async (url: string) => Response.json(routes[url] ?? { success: false }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ProviderSettings", () => {
  it("detects a local server and fills in provider, URL and models", async () => {
    mockFetch({ "/api/providers/discover": { success: true, data: [lmStudio] } });
    render(<Harness />);

    fireEvent.click(screen.getByRole("button", { name: "Detect local" }));
    fireEvent.click(await screen.findByRole("button", { name: /Use LM Studio at http:\/\/127.0.0.1:1235\/v1 \(2 models\)/ }));

    expect(latest.provider).toBe("lmstudio");
    expect(latest.baseUrl).toBe("http://127.0.0.1:1235/v1");
    expect(screen.getByText("Connected to LM Studio: 2 chat models available.")).toBeTruthy();
    // Capabilities are shown with each model; embeddings are not offered
    expect(screen.getAllByRole("option", { name: /Qwen2.5 Coder 7B \(no tool calling · 32k context\)/ })).toHaveLength(1);
    expect(screen.queryByRole("option", { name: /nomic-embed/ })).toBeNull();
  });

  it("offers only models that can see images as the vision model", async () => {
    mockFetch({ "/api/providers/test": { success: true, data: lmStudio } });
    render(<Harness initial={{ ...empty, provider: "lmstudio", baseUrl: "http://127.0.0.1:1235/v1" }} />);

    const vision = (await screen.findByText("Vision model")).parentElement!.querySelector("select")!;
    await waitFor(() => expect(vision.querySelectorAll("option")).toHaveLength(2));
    fireEvent.change(vision, { target: { value: "qwen2.5-vl-7b" } });
    expect(latest.visionModel).toBe("qwen2.5-vl-7b");
  });

  it("shows why a connection failed", async () => {
    mockFetch({ "/api/providers/test": { success: false, data: { ...lmStudio, ok: false, models: [],
      message: "Could not connect to http://localhost:11434/api/tags (ClientConnectorError)." } } });
    render(<Harness initial={{ ...empty, provider: "ollama" }} />);

    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    expect(await screen.findByText(/Could not connect/)).toBeTruthy();
  });

  it("switching provider replaces a default URL but keeps a custom one", () => {
    mockFetch({});
    render(<Harness initial={{ ...empty, baseUrl: "https://api.openai.com/v1" }} />);
    const provider = screen.getByText("Provider").parentElement!.querySelector("select")!;

    fireEvent.change(provider, { target: { value: "ollama" } });
    expect(latest.baseUrl).toBe("http://localhost:11434/v1");

    cleanup();
    render(<Harness initial={{ ...empty, baseUrl: "http://gpu-box:11434/v1" }} />);
    fireEvent.change(screen.getByText("Provider").parentElement!.querySelector("select")!, { target: { value: "ollama" } });
    expect(latest.baseUrl).toBe("http://gpu-box:11434/v1");
  });
});
