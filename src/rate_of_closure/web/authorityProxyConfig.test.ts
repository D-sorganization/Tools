import { describe, expect, it } from "vitest";

import { buildAuthorityProxyConfig } from "./authorityProxyConfig";

describe("authority Vite proxy configuration", () => {
  it("injects the ephemeral token at the dev server rather than browser code", () => {
    const proxy = buildAuthorityProxyConfig({
      ROC_AUTHORITY_URL: "http://127.0.0.1:54321",
      ROC_AUTHORITY_TOKEN: "test-ephemeral-token",
    });

    expect(proxy?.target).toBe("http://127.0.0.1:54321");
    expect(proxy?.headers).toEqual({ Authorization: "Bearer test-ephemeral-token" });
  });

  it("disables the proxy when no authority is launched", () => {
    expect(buildAuthorityProxyConfig({})).toBeUndefined();
  });

  it("rejects partial, remote, or whitespace-bearing configuration", () => {
    expect(() => buildAuthorityProxyConfig({
      ROC_AUTHORITY_URL: "http://127.0.0.1:54321",
    })).toThrow(/together/i);
    expect(() => buildAuthorityProxyConfig({
      ROC_AUTHORITY_URL: "https://example.com",
      ROC_AUTHORITY_TOKEN: "test-ephemeral-token",
    })).toThrow(/loopback/i);
    expect(() => buildAuthorityProxyConfig({
      ROC_AUTHORITY_URL: "http://127.0.0.1:54321",
      ROC_AUTHORITY_TOKEN: " token ",
    })).toThrow(/token/i);
  });
});
