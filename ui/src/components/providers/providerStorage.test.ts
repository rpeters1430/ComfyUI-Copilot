import { beforeEach, describe, expect, it } from "vitest";

import { detectProviderKind, loadProviderSettings, saveProviderSettings } from "./providerStorage";

beforeEach(() => localStorage.clear());

describe("detectProviderKind", () => {
  it.each([
    ["http://localhost:11434/v1", "ollama"],
    ["http://127.0.0.1:1234/v1", "lmstudio"],
    ["http://localhost:1235/v1", "lmstudio"],
    ["https://api.openai.com/v1", "openai"],
    ["not a url", "openai"],
  ])("%s -> %s", (url, kind) => {
    expect(detectProviderKind(url)).toBe(kind);
  });
});

describe("settings storage", () => {
  it("migrates installs that only saved a base URL", () => {
    localStorage.setItem("openaiBaseUrl", "http://localhost:11434/v1");
    localStorage.setItem("workflowLLMModel", "qwen3:14b");

    expect(loadProviderSettings()).toEqual({
      provider: "ollama",
      baseUrl: "http://localhost:11434/v1",
      apiKey: "",
      workflowModel: "qwen3:14b",
      visionModel: "",
      toolMode: "auto",
    });
  });

  it("round-trips and reports changes", () => {
    const settings = {
      provider: "lmstudio" as const, baseUrl: "http://localhost:1235/v1", apiKey: "",
      workflowModel: "", visionModel: "qwen2.5-vl-7b", toolMode: "json" as const,
    };
    expect(saveProviderSettings(settings)).toBe(true);
    expect(loadProviderSettings()).toEqual(settings);
    expect(saveProviderSettings(settings)).toBe(false);
    // Empty values are removed rather than stored as ""
    expect(localStorage.getItem("openaiApiKey")).toBeNull();
  });
});
