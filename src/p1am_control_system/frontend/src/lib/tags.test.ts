import { describe, it, expect } from "vitest";
import { TAG_COUNT, TAG_INDICES, tagName, parseTagId, isTagName } from "./tags";

describe("tags helpers", () => {
  it("exposes a consistent TAG_COUNT and index list", () => {
    expect(TAG_COUNT).toBe(32);
    expect(TAG_INDICES).toHaveLength(TAG_COUNT);
    expect(TAG_INDICES[0]).toBe(0);
    expect(TAG_INDICES[TAG_COUNT - 1]).toBe(TAG_COUNT - 1);
  });

  it("builds canonical tag names", () => {
    expect(tagName(0)).toBe("TAG_0");
    expect(tagName(31)).toBe("TAG_31");
  });

  it("parses valid in-range tag ids", () => {
    expect(parseTagId("TAG_0")).toBe(0);
    expect(parseTagId("TAG_31")).toBe(31);
  });

  it("rejects out-of-range, malformed, or non-tag names", () => {
    expect(parseTagId("TAG_32")).toBeNull();
    expect(parseTagId("TAG_-1")).toBeNull();
    expect(parseTagId("TAG_abc")).toBeNull();
    expect(parseTagId("FLOW_1")).toBeNull();
    expect(parseTagId("TAG_")).toBeNull();
  });

  it("isTagName mirrors parseTagId", () => {
    expect(isTagName("TAG_5")).toBe(true);
    expect(isTagName("TAG_99")).toBe(false);
  });
});
