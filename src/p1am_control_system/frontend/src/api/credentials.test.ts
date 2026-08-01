import { describe, expect, it, beforeEach, vi, afterEach } from "vitest";

/**
 * Credential handling for the HMI (issue #4007).
 *
 * The shipped HMI had none: no `X-API-Key`, no WebSocket credential, nothing.
 * That is why the production installer hardcoded `P1AM_DEV_NO_AUTH=1` — with
 * authentication on, the product did not work. These tests pin the contract
 * that makes an authenticated deployment usable.
 */

const STORAGE_KEY = "p1am.apiKey";
const TEST_KEY = "operator-key-under-test";

/** Re-import the module so its one-shot bootstrap runs again per test. */
async function freshModule() {
  vi.resetModules();
  return import("./credentials");
}

/**
 * A conforming `Storage` stand-in. The global `localStorage` this test runner
 * exposes is a bare object with no `getItem`/`setItem`, so install a real one
 * to exercise the persistence path rather than the module's in-memory
 * fallback.
 */
class MemoryStorage {
  private readonly entries = new Map<string, string>();
  getItem(key: string): string | null {
    return this.entries.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.entries.set(key, value);
  }
  removeItem(key: string): void {
    this.entries.delete(key);
  }
  clear(): void {
    this.entries.clear();
  }
}

beforeEach(() => {
  Object.defineProperty(window, "localStorage", {
    value: new MemoryStorage(),
    configurable: true,
    writable: true,
  });
  window.history.replaceState(null, "", "/");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("credential storage", () => {
  it("returns null when nothing has been provisioned", async () => {
    const { getApiKey } = await freshModule();
    expect(getApiKey()).toBeNull();
  });

  it("round-trips a key through localStorage", async () => {
    const { getApiKey, setApiKey } = await freshModule();
    setApiKey(TEST_KEY);
    expect(getApiKey()).toBe(TEST_KEY);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(TEST_KEY);
  });

  it("trims surrounding whitespace from a pasted key", async () => {
    const { getApiKey, setApiKey } = await freshModule();
    setApiKey(`  ${TEST_KEY}\n`);
    expect(getApiKey()).toBe(TEST_KEY);
  });

  it("treats a blank key as a clear", async () => {
    const { getApiKey, setApiKey } = await freshModule();
    setApiKey(TEST_KEY);
    setApiKey("   ");
    expect(getApiKey()).toBeNull();
  });

  it("clears the stored key", async () => {
    const { clearApiKey, getApiKey, setApiKey } = await freshModule();
    setApiKey(TEST_KEY);
    clearApiKey();
    expect(getApiKey()).toBeNull();
  });
});

describe("kiosk bootstrap from the URL fragment", () => {
  it("adopts a key seeded in the fragment", async () => {
    window.history.replaceState(null, "", `/#apikey=${TEST_KEY}`);
    const { getApiKey } = await freshModule();
    expect(getApiKey()).toBe(TEST_KEY);
  });

  it("scrubs the key out of the address bar immediately", async () => {
    window.history.replaceState(null, "", `/#apikey=${TEST_KEY}`);
    const { getApiKey } = await freshModule();
    getApiKey();
    expect(window.location.hash).not.toContain(TEST_KEY);
    expect(window.location.hash).not.toContain("apikey");
  });

  it("leaves an unrelated fragment intact", async () => {
    window.history.replaceState(null, "", `/#tab=trends&apikey=${TEST_KEY}`);
    const { getApiKey } = await freshModule();
    getApiKey();
    expect(window.location.hash).toContain("tab=trends");
    expect(window.location.hash).not.toContain(TEST_KEY);
  });
});

describe("authHeaders", () => {
  it("always sends the preflight-forcing marker", async () => {
    const { authHeaders } = await freshModule();
    expect(authHeaders()["X-Requested-With"]).toBe("p1am-hmi");
  });

  it("omits the credential header when no key is stored", async () => {
    const { authHeaders } = await freshModule();
    expect(authHeaders()["X-API-Key"]).toBeUndefined();
  });

  it("sends the credential header once a key is stored", async () => {
    const { authHeaders, setApiKey } = await freshModule();
    setApiKey(TEST_KEY);
    expect(authHeaders()["X-API-Key"]).toBe(TEST_KEY);
  });
});

describe("promptForApiKey", () => {
  it("stores what the operator enters", async () => {
    const { getApiKey, promptForApiKey } = await freshModule();
    vi.spyOn(window, "prompt").mockReturnValue(TEST_KEY);
    expect(promptForApiKey()).toBe(TEST_KEY);
    expect(getApiKey()).toBe(TEST_KEY);
  });

  it("leaves the stored key alone when dismissed", async () => {
    const { getApiKey, promptForApiKey, setApiKey } = await freshModule();
    setApiKey(TEST_KEY);
    vi.spyOn(window, "prompt").mockReturnValue(null);
    expect(promptForApiKey()).toBeNull();
    expect(getApiKey()).toBe(TEST_KEY);
  });
});

describe("apiFetch integration", () => {
  it("attaches the credential to every backend call", async () => {
    vi.resetModules();
    const { setApiKey } = await import("./credentials");
    const { apiFetch } = await import("./client");
    setApiKey(TEST_KEY);

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/routing");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = new Headers(init.headers);
    expect(headers.get("X-API-Key")).toBe(TEST_KEY);
    expect(headers.get("X-Requested-With")).toBe("p1am-hmi");
    vi.unstubAllGlobals();
  });

  it("lets an explicit caller header win", async () => {
    vi.resetModules();
    const { setApiKey } = await import("./credentials");
    const { apiFetch } = await import("./client");
    setApiKey(TEST_KEY);

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/routing", { headers: { "X-API-Key": "override" } });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("X-API-Key")).toBe("override");
    vi.unstubAllGlobals();
  });
});
