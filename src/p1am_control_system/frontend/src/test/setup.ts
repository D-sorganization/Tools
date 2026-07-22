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

// Unmount React trees after every test so the jsdom document stays clean.
afterEach(() => {
  cleanup();
});
