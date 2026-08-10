import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type SyntheticEvent,
} from "react";

interface HorizontalBounds {
  readonly left: number;
  readonly right: number;
}

const DEFAULT_VIEWPORT_GUTTER_PX = 16;

/** Return the horizontal translation needed to keep a bounded popover on-screen. */
export function horizontalViewportOffset(
  bounds: HorizontalBounds,
  viewportWidth: number,
  gutterPx = DEFAULT_VIEWPORT_GUTTER_PX,
): number {
  if (![bounds.left, bounds.right, viewportWidth, gutterPx].every(Number.isFinite)) {
    throw new RangeError("popover bounds, viewport width, and gutter must be finite");
  }
  if (bounds.right < bounds.left || viewportWidth <= 0 || gutterPx < 0) {
    throw new RangeError("popover bounds and viewport dimensions are invalid");
  }
  const width = bounds.right - bounds.left;
  const maximumLeft = Math.max(gutterPx, viewportWidth - gutterPx - width);
  const clampedLeft = Math.min(Math.max(bounds.left, gutterPx), maximumLeft);
  return clampedLeft - bounds.left;
}

/** Keep an uncontrolled native details popover inside the current viewport. */
export function useViewportClampedPopover() {
  const panelRef = useRef<HTMLDivElement>(null);
  const offsetRef = useRef(0);
  const [offsetPx, setOffsetPx] = useState(0);

  const updateOffset = useCallback(() => {
    const panel = panelRef.current;
    const details = panel?.closest("details");
    if (!panel || !(details instanceof HTMLDetailsElement) || !details.open) return;
    const bounds = panel.getBoundingClientRect();
    const baseBounds = {
      left: bounds.left - offsetRef.current,
      right: bounds.right - offsetRef.current,
    };
    const nextOffset = horizontalViewportOffset(
      baseBounds,
      document.documentElement.clientWidth,
    );
    offsetRef.current = nextOffset;
    setOffsetPx(nextOffset);
  }, []);

  useEffect(() => {
    window.addEventListener("resize", updateOffset);
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(updateOffset);
    if (panelRef.current) observer?.observe(panelRef.current);
    return () => {
      window.removeEventListener("resize", updateOffset);
      observer?.disconnect();
    };
  }, [updateOffset]);

  const onToggle = useCallback((event: SyntheticEvent<HTMLDetailsElement>) => {
    if (event.currentTarget.open) {
      updateOffset();
      return;
    }
    offsetRef.current = 0;
    setOffsetPx(0);
  }, [updateOffset]);

  const style: CSSProperties | undefined = offsetPx === 0
    ? undefined
    : { transform: `translateX(${offsetPx}px)` };
  return { onToggle, panelRef, style } as const;
}
