import { describe, expect, it } from "vitest";

import {
  LOCAL_COMPANION_WEB_RUNTIME,
  parseEmbeddedWebRuntime,
  parseWebRuntime,
  STATIC_INSPECTION_WEB_RUNTIME,
  WEB_RUNTIME_SCHEMA,
  WEB_RUNTIME_ELEMENT_ID,
} from "./webRuntime";

describe("Rate of Closure web runtime descriptor", () => {
  it("accepts and freezes the exact static-inspection descriptor", () => {
    const runtime = parseWebRuntime({
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "static_inspection",
      release_revision: "development",
    });

    expect(runtime).toEqual(STATIC_INSPECTION_WEB_RUNTIME);
    expect(Object.isFrozen(runtime)).toBe(true);
  });

  it("accepts only the fixed same-origin API path in companion mode", () => {
    expect(parseWebRuntime({
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "local_companion",
      release_revision: "0123456789abcdef0123456789abcdef01234567",
      authority_path: "/api/rate-of-closure/v1",
    })).toEqual({
      ...LOCAL_COMPANION_WEB_RUNTIME,
      release_revision: "0123456789abcdef0123456789abcdef01234567",
    });
    expect(() => parseWebRuntime({
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "local_companion",
      release_revision: "development",
      authority_path: "http://127.0.0.1:8912/api/rate-of-closure/v1",
    })).toThrow(/authority_path/i);
  });

  it.each([
    null,
    {},
    {
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "static_inspection",
      release_revision: "development",
      authority_path: "/api/rate-of-closure/v1",
    },
    {
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "local_companion",
      release_revision: "development",
    },
    {
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: "static_inspection",
      release_revision: "main",
    },
    {
      schema_version: "rate-of-closure/web-runtime/v2",
      mode: "static_inspection",
      release_revision: "development",
    },
  ])("rejects malformed, ambiguous, or extra-field descriptors", (value) => {
    expect(() => parseWebRuntime(value)).toThrow();
  });

  it("reads exactly one strict embedded production descriptor", () => {
    document.body.innerHTML = `<script id="${WEB_RUNTIME_ELEMENT_ID}" type="application/json">${JSON.stringify(STATIC_INSPECTION_WEB_RUNTIME)}</script>`;
    expect(parseEmbeddedWebRuntime(document)).toEqual(STATIC_INSPECTION_WEB_RUNTIME);
    document.body.insertAdjacentHTML("beforeend", document.body.innerHTML);
    expect(() => parseEmbeddedWebRuntime(document)).toThrow(/exactly one/i);
  });

  it.each([
    "<div></div>",
    `<script id="${WEB_RUNTIME_ELEMENT_ID}" type="text/plain">{}</script>`,
    `<script id="${WEB_RUNTIME_ELEMENT_ID}" type="application/json">{bad}</script>`,
  ])("fails closed on an invalid embedded descriptor", (markup) => {
    document.body.innerHTML = markup;
    expect(() => parseEmbeddedWebRuntime(document)).toThrow();
  });
});
