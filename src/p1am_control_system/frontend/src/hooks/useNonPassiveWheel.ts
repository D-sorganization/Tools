import { useEffect, useRef, type RefObject } from "react";

/**
 * Attach a NON-passive `wheel` listener so the handler's `preventDefault()`
 * actually suppresses page scroll.
 *
 * React 18 registers its synthetic `onWheel` as a *passive* listener at the
 * document root, so calling `preventDefault()` there is a no-op (and logs a
 * warning) — wheel-zooming a chart would also scroll the page. One shared,
 * tested attach point for every trend keeps that fix DRY. The handler is held in
 * a ref so the latest closure runs without re-binding the listener each render.
 */
export function useNonPassiveWheel<T extends Element>(
  ref: RefObject<T | null>,
  handler: (e: WheelEvent) => void,
): void {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Typed as EventListener (a generic element's addEventListener resolves to
    // the string-type overload); the event is always a WheelEvent for "wheel".
    const listener: EventListener = (e) => handlerRef.current(e as WheelEvent);
    el.addEventListener("wheel", listener, { passive: false });
    return () => el.removeEventListener("wheel", listener);
  }, [ref]);
}
