import type { VariationMode } from "../model/variation";

export {
  defaultVariationPlan,
  defaultVariationSpec as defaultSpec,
} from "../model/variationDefaults";

export const MODE_LABELS: Record<VariationMode, string> = {
  delivery: "Delivery → Impact → Flight",
  swing: "Pendulum Swing → Impact → Flight",
  launch: "Launch Conditions → Flight",
};

export const PANEL_CLASS =
  "rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur";
export const INPUT_CLASS =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none";
export const BUTTON_CLASS =
  "rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 transition-colors hover:border-slate-500 disabled:opacity-40";

export const downloadText = (
  name: string,
  text: string,
  type: string,
): void => {
  downloadBlob(name, new Blob([text], { type }));
};

export const downloadBlob = (name: string, blob: Blob): void => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
};

export const downloadSvgElement = (
  name: string,
  element: SVGSVGElement,
): void => {
  const source = new XMLSerializer().serializeToString(element);
  downloadText(name, source, "image/svg+xml;charset=utf-8");
};

export const sensitivityHeat = (fraction: number): string => {
  const bounded = Math.min(Math.max(fraction, 0), 1);
  const mix = (start: number, end: number) =>
    Math.round(start + bounded * (end - start));
  return `rgb(${mix(37, 235)}, ${mix(66, 106)}, ${mix(96, 60)})`;
};
