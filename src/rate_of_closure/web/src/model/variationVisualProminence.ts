import {
  visualizationReferenceEnvironments,
  visualizationTabs,
} from "./visualizationTabManifest";

interface ViewportSize { readonly width: number; readonly height: number }
interface RevealOptions {
  readonly viewport: ViewportSize;
  readonly reducedMotion: boolean;
}
interface RevealWindow {
  readonly innerWidth: number;
  readonly innerHeight: number;
  requestAnimationFrame(callback: FrameRequestCallback): number;
  matchMedia?(query: string): Pick<MediaQueryList, "matches">;
}
interface VisibleRectangle { x: number; y: number; width: number; height: number }

const clipped = (value: VisibleRectangle, clip: DOMRect): VisibleRectangle => {
  const left = Math.max(value.x, clip.left);
  const top = Math.max(value.y, clip.top);
  const right = Math.min(value.x + value.width, clip.right);
  const bottom = Math.min(value.y + value.height, clip.bottom);
  return { x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
};

export const ancestorClippedIntersection = (
  element: HTMLElement,
  viewport: ViewportSize,
): VisibleRectangle => {
  const initial = element.getBoundingClientRect();
  let result: VisibleRectangle = {
    x: initial.left, y: initial.top, width: initial.width, height: initial.height,
  };
  let ancestor = element.parentElement;
  while (ancestor !== null) {
    const style = getComputedStyle(ancestor);
    if ([style.overflow, style.overflowX, style.overflowY]
      .some((value) => ["hidden", "clip", "scroll", "auto"].includes(value))) {
      result = clipped(result, ancestor.getBoundingClientRect());
    }
    ancestor = ancestor.parentElement;
  }
  return clipped(result, new DOMRect(0, 0, viewport.width, viewport.height));
};

const thresholds = (viewport: ViewportSize): { width: number; height: number } => {
  const authority = visualizationReferenceEnvironments.react;
  const variation = visualizationTabs("react").find((entry) => entry.tabId === "variation");
  if (variation === undefined) throw new Error("Variation visual authority is missing");
  const responsive = viewport.width <= authority.additionalViewportsPx
    .reduce((smallest, item) => Math.min(smallest, item[0]), Number.POSITIVE_INFINITY);
  return responsive
    ? { width: authority.responsiveMinimumVisibleWidthPx,
      height: authority.responsiveMinimumVisibleHeightPx }
    : { width: authority.minimumVisibleWidthPx, height: variation.minimumVisibleHeightPx };
};

export const revealMeaningfulVisual = (
  element: HTMLElement,
  options: RevealOptions,
): boolean => {
  const visible = ancestorClippedIntersection(element, options.viewport);
  const minimum = thresholds(options.viewport);
  if (visible.width >= minimum.width && visible.height >= minimum.height) return false;
  if (typeof element.scrollIntoView !== "function") return false;
  element.scrollIntoView({
    behavior: options.reducedMotion ? "auto" : "smooth",
    block: "nearest",
    inline: "nearest",
  });
  return true;
};

export const scheduleMeaningfulVisualReveal = (
  target: () => HTMLElement | null,
  owner: RevealWindow = window,
): number => owner.requestAnimationFrame(() => {
  const element = target();
  if (element === null) return;
  revealMeaningfulVisual(element, {
    viewport: { width: owner.innerWidth, height: owner.innerHeight },
    reducedMotion: owner.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  });
});
