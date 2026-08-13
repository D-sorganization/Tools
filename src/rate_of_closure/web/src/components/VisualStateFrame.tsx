import type { ReactNode } from "react";

import type { VariationVisualState } from "../model/variationVisualState";

interface VisualStateFrameProps {
  state: VariationVisualState;
  announcement: string;
  children: ReactNode;
}

export function VisualStateFrame({
  state,
  announcement,
  children,
}: VisualStateFrameProps): JSX.Element {
  const retained = state.visualOrigin === "prior-accepted";
  const label = retained
    ? "Variation visualization — prior accepted result retained"
    : state.visualOrigin === "current-accepted"
      ? "Variation visualization — current accepted result"
      : "Variation visualization — analysis preview";
  return (
    <div aria-label={label} className="min-w-0" data-phase={state.phase}
      data-visual-origin={state.visualOrigin}>
      {state.phase !== "empty" && state.phase !== "result" && (
        <div aria-label="Variation visualization state"
          className={`mb-3 w-full rounded-lg border px-3 py-2 text-xs ${
            state.phase === "error"
              ? "border-rose-400/60 bg-rose-950/40 text-rose-100"
              : "border-sky-400/50 bg-sky-950/30 text-sky-100"
          }`}>
          {retained ? `Prior accepted result retained. ${announcement}` : announcement}
        </div>
      )}
      {children}
    </div>
  );
}
