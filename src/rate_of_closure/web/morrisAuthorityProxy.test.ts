/** Server-only authority proxy configuration tests. */

import { describe, expect, it } from "vitest";

import { morrisAuthorityProxy } from "./morrisAuthorityProxy";

describe("Morris authority Vite proxy", () => {
  it("injects the bearer only in the server proxy", () => {
    const proxy = morrisAuthorityProxy({
      ROC_MORRIS_AUTHORITY_URL: "http://127.0.0.1:43210",
      ROC_MORRIS_AUTHORITY_TOKEN: "server-secret",
    });
    expect(proxy["/api/rate-of-closure"]).toMatchObject({
      target: "http://127.0.0.1:43210",
      headers: { Authorization: "Bearer server-secret" },
    });
    expect(JSON.stringify(proxy)).not.toContain("VITE_");
  });

  it("is absent without runtime vars and fails closed on partial/nonloopback config", () => {
    expect(morrisAuthorityProxy({})).toEqual({});
    expect(() => morrisAuthorityProxy({ ROC_MORRIS_AUTHORITY_TOKEN: "secret" })).toThrow();
  });

  it.each([
    "http://0.0.0.0:8000",
    `http://user:${"password"}@127.0.0.1:8000`,
    "http://127.0.0.1:8000/path",
    "http://127.0.0.1:8000?query=yes",
    "http://127.0.0.1:8000#fragment",
    "http://127.0.0.1:99999",
    "http://127.0.0.1:8000\r\nheader:value",
  ])("rejects unsafe target %s", (target) => {
    expect(() => morrisAuthorityProxy({
      ROC_MORRIS_AUTHORITY_URL: target,
      ROC_MORRIS_AUTHORITY_TOKEN: "server-secret",
    })).toThrow();
  });

  it.each([" short ", "bad\r\ntoken", "seven77"])("rejects unsafe token %s", (token) => {
    expect(() => morrisAuthorityProxy({
      ROC_MORRIS_AUTHORITY_URL: "http://127.0.0.1:8000",
      ROC_MORRIS_AUTHORITY_TOKEN: token,
    })).toThrow("token");
  });
});
