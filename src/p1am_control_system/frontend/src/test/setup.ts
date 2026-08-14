import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement PointerEvent, so React's synthetic onPointerDown/
// Move/Up/Leave never fire under fireEvent.pointer*. Alias it to MouseEvent
// (which carries clientX/clientY) so the pointer-driven trend interactions —
// pan/zoom drag-select and the hover crosshair — are testable. No-op in a real
// browser where PointerEvent already exists.
if (typeof window !== "undefined" && typeof window.PointerEvent === "undefined") {
  class PointerEventPolyfill extends MouseEvent {
    readonly pointerId: number;
    constructor(type: string, params: PointerEventInit = {}) {
      super(type, params);
      this.pointerId = params.pointerId ?? 0;
    }
  }
  window.PointerEvent = PointerEventPolyfill as unknown as typeof PointerEvent;
}

// Node 22+ defines `localStorage`/`sessionStorage` on globalThis, and vitest's
// jsdom environment only copies a jsdom global over an already-present Node
// global when the key appears in its own curated key list — which these two do
// not. So jsdom's working Storage is never installed and Node's wins. Node's
// `sessionStorage` is in-memory and behaves; its `localStorage` is backed by
// `--localstorage-file`, and with no valid path it degrades to a bare object
// with no getItem/setItem/removeItem/clear — hence "localStorage.clear is not a
// function". The HMI genuinely persists operator layout/theme in Web Storage,
// so every suite needs a working Storage regardless of host Node version.
//
// Installed unconditionally rather than behind a conformance probe: this file
// only ever runs under the test runner, a fresh in-memory Storage per test file
// is the isolation we want anyway, and merely READING Node's lazy `localStorage`
// to probe it fires a `--localstorage-file` warning in every worker.

/**
 * A spec-shaped in-memory Storage: keys and values are coerced to strings,
 * `getItem` yields null for absent keys, and `key(n)` follows insertion order.
 * The one departure from the real thing is named property access
 * (`localStorage.foo = "bar"`), which needs a Proxy and which nothing in this
 * app uses — every call site goes through the documented methods.
 */
function createStorage(): Storage {
  const entries = new Map<string, string>();
  return {
    get length(): number {
      return entries.size;
    },
    key(index: number): string | null {
      const at = Math.trunc(index);
      if (!Number.isFinite(at) || at < 0 || at >= entries.size) return null;
      return [...entries.keys()][at];
    },
    getItem(key: string): string | null {
      const value = entries.get(String(key));
      return value === undefined ? null : value;
    },
    setItem(key: string, value: string): void {
      entries.set(String(key), String(value));
    },
    removeItem(key: string): void {
      entries.delete(String(key));
    },
    clear(): void {
      entries.clear();
    },
  };
}

// Install on BOTH `globalThis` and `window`: under jsdom a component reaching
// for a bare `localStorage` resolves it on `window`, so defining only the
// globalThis property leaves `window.localStorage` as Node's broken object.
// One shared instance so both names see the same entries.
const memoryStorage = createStorage();
for (const target of [globalThis, window] as unknown as Record<string, unknown>[]) {
  Object.defineProperty(target, "localStorage", {
    value: memoryStorage,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}

// Unmount React trees after every test so the jsdom document stays clean.
afterEach(() => {
  cleanup();
});
