/**
 * Test-only helpers for driving the plot hover crosshair under jsdom.
 *
 * jsdom has no `PointerEvent` constructor and `getBoundingClientRect` reports a
 * zero-sized box, so hover tests must (1) mock the SVG's client rect so the
 * cursor→pixel math is exact and (2) dispatch `pointermove` as a `MouseEvent`
 * (which carries `clientX`, unlike the generic-Event fallback). Centralised here
 * so every plot's hover test drives the cursor identically (DRY).
 *
 * Not a spec file — deliberately named without `.test` so Vitest doesn't collect
 * it; it is imported by the plot `*.test.tsx` suites.
 */

import { fireEvent } from "@testing-library/react";

/** Force a deterministic client rect (origin 0,0; size `width`×`height`). */
export function mockSvgRect(
  svg: SVGSVGElement,
  width: number,
  height: number,
): void {
  svg.getBoundingClientRect = () =>
    ({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: width,
      bottom: height,
      width,
      height,
      toJSON: () => ({}),
    }) as DOMRect;
}

/** Dispatch a `pointermove` at `clientX` (a MouseEvent carries `clientX`). */
export function pointerMoveAt(svg: SVGSVGElement, clientX: number): void {
  fireEvent(svg, new MouseEvent("pointermove", { clientX, bubbles: true }));
}

/** Dispatch a `pointerleave` on the element. */
export function pointerLeaveSvg(svg: SVGSVGElement): void {
  fireEvent.pointerLeave(svg);
}
