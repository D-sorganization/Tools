/** Finite pointer coordinates for browsers and reduced DOM test environments. */

export interface PointerCoordinates {
  readonly x: number;
  readonly y: number;
}

export function pointerCoordinates(
  event: Pick<PointerEvent, "clientX" | "clientY">,
): PointerCoordinates {
  return {
    x: Number.isFinite(event.clientX) ? event.clientX : 0,
    y: Number.isFinite(event.clientY) ? event.clientY : 0,
  };
}
