import { describe, expect, it } from "vitest";

import { readFileAsDataUrl } from "./fileUtils";

describe("readFileAsDataUrl", () => {
  it("encodes a file as a data URL", async () => {
    const file = new File(["hi"], "a.png", { type: "image/png" });
    expect(await readFileAsDataUrl(file)).toBe("data:image/png;base64,aGk=");
  });
});
