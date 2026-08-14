import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Node 25 exposes an incomplete global localStorage unless it receives a
// backing-file option. Install an isolated standards-shaped test store when
// that host object is unusable; production browsers retain their native store.
if (
  typeof window !== "undefined" &&
  (typeof globalThis.localStorage?.getItem !== "function" ||
    typeof globalThis.localStorage?.clear !== "function")
) {
  const values = new Map<string, string>();
  const testStorage: Storage = {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => {
      values.delete(key);
    },
    setItem: (key, value) => {
      values.set(key, String(value));
    },
  };
  const descriptor = {
    configurable: true,
    value: testStorage,
  };
  Object.defineProperty(globalThis, "localStorage", descriptor);
  Object.defineProperty(window, "localStorage", descriptor);
}

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

// Unmount React trees after every test so the jsdom document stays clean.
afterEach(() => {
  cleanup();
});
