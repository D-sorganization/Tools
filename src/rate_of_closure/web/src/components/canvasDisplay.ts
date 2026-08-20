import type { RefObject } from "react";

export interface LogicalCanvasSize {
  readonly width: number;
  readonly height: number;
}

/** Prepare a responsive canvas backing store for the current CSS size and DPR. */
export function canvasContext(
  canvas: HTMLCanvasElement,
  logicalSize: LogicalCanvasSize,
): CanvasRenderingContext2D | null {
  const bounds = canvas.getBoundingClientRect();
  const cssWidth = bounds.width > 0 ? bounds.width : logicalSize.width;
  const cssHeight = bounds.height > 0
    ? bounds.height
    : cssWidth * logicalSize.height / logicalSize.width;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const backingWidth = Math.max(1, Math.round(cssWidth * dpr));
  const backingHeight = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
  const context = canvas.getContext("2d");
  context?.setTransform(
    backingWidth / logicalSize.width, 0, 0,
    backingHeight / logicalSize.height, 0, 0,
  );
  return context;
}

/** Redraw on element resize and window resize (including browser DPR changes). */
export function observeCanvas(
  reference: RefObject<HTMLCanvasElement | null>,
  draw: () => void,
): () => void {
  const canvas = reference.current;
  if (!canvas) return () => undefined;
  const observer = typeof ResizeObserver === "undefined"
    ? null
    : new ResizeObserver(draw);
  observer?.observe(canvas);
  window.addEventListener("resize", draw);
  draw();
  return () => {
    observer?.disconnect();
    window.removeEventListener("resize", draw);
  };
}
