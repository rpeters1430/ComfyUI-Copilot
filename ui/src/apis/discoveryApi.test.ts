import { afterEach, describe, expect, it, vi } from "vitest";
import { getLibraryWorkflow } from "./discoveryApi";

afterEach(() => vi.restoreAllMocks());

describe("getLibraryWorkflow", () => {
  it("loads a library workflow by id from the local backend", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ success: true, data: { id: "abc", workflow: { nodes: [] } } })),
    );
    const result = await getLibraryWorkflow("abc/../x");
    expect(fetchMock).toHaveBeenCalledWith("/api/discovery/workflows/abc%2F..%2Fx");
    expect(result.workflow).toEqual({ nodes: [] });
  });

  it("surfaces the backend's message when the workflow is gone", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ success: false, message: "Workflow not found" }), { status: 404 }),
    );
    await expect(getLibraryWorkflow("gone")).rejects.toThrow("Workflow not found");
  });
});
