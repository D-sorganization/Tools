import {
  keysForMode,
  localizedTorqueJointId,
  variableDef,
  type NoiseSpecTs,
  type VariationMode,
  type VariationPlanTs,
} from "../model/variation";
import type { BallSetup } from "../model/ballSetup";

export const MODE_LABELS: Record<VariationMode, string> = {
  delivery: "Delivery → Impact → Flight",
  swing: "Pendulum Swing → Impact → Flight",
  launch: "Launch Conditions → Flight",
};

export const PANEL_CLASS =
  "rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur";
export const INPUT_CLASS =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";
export const BUTTON_CLASS =
  "rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 transition-colors hover:border-slate-500 disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

export const defaultSpec = (
  mode: VariationMode,
  excluded: ReadonlySet<string> = new Set(),
  ballSetup?: BallSetup,
): NoiseSpecTs => {
  const keys = keysForMode(mode, ballSetup);
  const key = keys.find((candidate) => !excluded.has(candidate)) ?? keys[0];
  const localizedJoint = localizedTorqueJointId(key);
  return {
    variableKey: key,
    distribution: "normal",
    scale: variableDef(key)?.typicalScale ?? 1,
    lower: null,
    upper: null,
    specId: key,
    timeWindowS: localizedJoint === null ? null : [0, 0.1],
    pointIds: localizedJoint === null ? [] : [localizedJoint],
  };
};

export const defaultVariationPlan = (): VariationPlanTs => ({
  mode: "delivery",
  baseVariables: {},
  noise: [defaultSpec("delivery")],
  nRuns: 200,
  seed: 0,
  flightModel: "waterloo_penner",
  groups: [],
});

export const downloadText = (name: string, text: string, type: string): void => {
  downloadBlob(name, new Blob([text], { type }));
};

export const readFileText = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result ?? "")));
    reader.addEventListener("error", () => reject(reader.error ?? new Error("File read failed")));
    reader.readAsText(file);
  });

export const downloadBlob = (name: string, blob: Blob): void => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
};

export const downloadSvgElement = (name: string, element: SVGSVGElement): void => {
  const source = new XMLSerializer().serializeToString(element);
  downloadText(name, source, "image/svg+xml;charset=utf-8");
};

export const sensitivityHeat = (fraction: number): string => {
  const bounded = Math.min(Math.max(fraction, 0), 1);
  const mix = (start: number, end: number) => Math.round(start + bounded * (end - start));
  return `rgb(${mix(37, 235)}, ${mix(66, 106)}, ${mix(96, 60)})`;
};
