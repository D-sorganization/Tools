import { describe, expect, it } from "vitest";

import { horizontalViewportOffset } from "./useViewportClampedPopover";

describe("horizontalViewportOffset", () => {
  it("moves a right-edge collision inside a 520 px viewport gutter", () => {
    expect(horizontalViewportOffset({ left: 366.5, right: 622.5 }, 520, 16))
      .toBe(-118.5);
  });

  it("preserves desktop placement when the popover already fits", () => {
    expect(horizontalViewportOffset({ left: 440, right: 696 }, 1440, 16)).toBe(0);
  });

  it("moves a left-edge collision inside the viewport gutter", () => {
    expect(horizontalViewportOffset({ left: -8, right: 248 }, 520, 16)).toBe(24);
  });
});
