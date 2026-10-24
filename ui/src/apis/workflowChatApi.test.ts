import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Builds without VITE_API_BASE_URL have no hosted server: everything must go to the local backend
vi.mock("../config", () => ({
  config: { apiBaseUrl: "https://remote.test", remoteServerEnabled: false },
}));

vi.mock("../utils/comfyapp", () => ({
  app: {
    extensionManager: { setting: { get: () => "en" } },
    graphToPrompt: async () => ({ output: {}, workflow: {} }),
  },
  api: { fetchApi: vi.fn() },
}));

import { WorkflowChatAPI } from "./workflowChatApi";

const finalLine = JSON.stringify({
  session_id: "s", text: "hi", finished: true, type: "message", format: "markdown", ext: null,
});

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  localStorage.clear();
  fetchMock = vi.fn(async () => new Response(finalLine + "\n"));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function drain<T>(stream: AsyncGenerator<T>) {
  for await (const _ of stream) {
    // consume
  }
}

function requestTo(path: string) {
  const call = fetchMock.mock.calls.find(([url]) => url === path);
  expect(call, `expected a request to ${path}`).toBeDefined();
  const init = call![1] as RequestInit;
  return { headers: init.headers as Record<string, string>, body: JSON.parse(init.body as string) };
}

describe("without a hosted server", () => {
  it("sends chat to the local backend with the base URL, even without an API key", async () => {
    // A keyless local server such as Ollama or LMStudio
    localStorage.setItem("openaiBaseUrl", "http://localhost:11434/v1");

    await drain(WorkflowChatAPI.streamInvokeServer("s", "hello"));

    const { headers } = requestTo("/api/chat/invoke");
    expect(headers["Openai-Base-Url"]).toBe("http://localhost:11434/v1");
    expect(headers["Openai-Api-Key"]).toBeUndefined();
  });

  it("sends intent requests (node/model search) to the local backend too", async () => {
    localStorage.setItem("openaiApiKey", "sk-user");

    await drain(WorkflowChatAPI.streamInvokeServer("s", "KSampler", [], "node_search"));

    const { headers } = requestTo("/api/chat/invoke");
    expect(headers["Openai-Api-Key"]).toBe("sk-user");
  });

  it("never contacts the remote server", async () => {
    localStorage.setItem("openaiApiKey", "sk-user");

    await drain(WorkflowChatAPI.streamInvokeServer("s", "hello"));
    await WorkflowChatAPI.trackEvent({ event_type: "x", message_type: "y" } as never);
    expect(await WorkflowChatAPI.fetchAnnouncement()).toBe("");

    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls.some((url) => url.startsWith("https://remote.test"))).toBe(false);
  });

  it("sends the selected model and base URL with debug requests", async () => {
    localStorage.setItem("openaiBaseUrl", "http://localhost:1234/v1");
    localStorage.setItem("models_selected", "qwen3-8b");

    await drain(WorkflowChatAPI.streamDebugAgent({ output: { "1": {} } }));

    const { headers, body } = requestTo("/api/debug-agent");
    expect(headers["Openai-Base-Url"]).toBe("http://localhost:1234/v1");
    expect(body.model_select).toBe("qwen3-8b");
  });

  it("generates GenLab prompts with the local backend", async () => {
    fetchMock.mockImplementationOnce(async () => Response.json({ success: true, data: ["a", "b"] }));
    localStorage.setItem("openaiBaseUrl", "http://localhost:1234/v1");

    expect(await WorkflowChatAPI.generateSDPrompts("a cat")).toEqual(["a", "b"]);
    const { headers } = requestTo("/api/param_debug/generate_sd_prompts");
    expect(headers["Openai-Base-Url"]).toBe("http://localhost:1234/v1");
  });
});
