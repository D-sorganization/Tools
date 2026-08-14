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

// Web Storage: Node >= 22 exposes its own experimental `localStorage` global.
// Because that global already exists on `globalThis`, vitest's jsdom
// environment does not install jsdom's Storage over it, and the object Node
// hands back without `--localstorage-file` implements NO Storage methods — so
// `localStorage.clear()` throws and every layout-persistence suite fails on a
// developer machine running a modern Node. The HMI genuinely persists operator
// layout/theme in Web Storage, so the tests must have a working Storage
// regardless of host Node version: install a spec-shaped in-memory one.
function installMemoryStorage(): void {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key: string) => (store.has(key) ? (store.get(key) as string) : null),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    removeItem: (key: string) => void store.delete(key),
    setItem: (key: string, value: string) => void store.set(String(key), String(value)),
  };
  for (const target of [globalThis, window] as unknown as Record<string, unknown>[]) {
    Object.defineProperty(target, "localStorage", {
      value: storage,
      configurable: true,
      writable: true,
    });
  }
}

if (typeof localStorage === "undefined" || typeof localStorage.clear !== "function") {
  installMemoryStorage();
}

// Unmount React trees after every test so the jsdom document stays clean.
afterEach(() => {
  cleanup();
});
