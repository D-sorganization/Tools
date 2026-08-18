import type { WindComparisonTs } from "../model/flightExplorer";
import { SPEED_UNITS as CANONICAL_SPEED_UNITS } from "../model/units";

export const SPEED_UNITS = CANONICAL_SPEED_UNITS;

export function boundedFlightError(error: unknown, retained: boolean): string {
  const message = error instanceof Error ? error.message : String(error);
  const normalized = [...message].map((character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f)
      ? " " : character;
  }).join("").trim();
  const suffix = retained
    ? "The prior accepted flight remains displayed."
    : "No accepted flight is available.";
  const diagnostic = normalized || "Flight computation failed";
  return `${diagnostic.slice(0, 510 - suffix.length)}. ${suffix}`;
}

interface DirectionConventionOption {
  value: string;
  label: string;
  disabled?: boolean;
  title?: string;
}

export const DIRECTION_CONVENTIONS: readonly DirectionConventionOption[] = [
  { value: "app_native", label: "App Native (+ Right)" },
  { value: "trackman_comparable", label: "TrackMan-Comparable (+ Right)" },
  {
    value: "foresight_comparable",
    label: "Foresight-Comparable (Sign Unavailable)",
    disabled: true,
    title: "Unavailable: the general public sign convention is not established independently of player handedness.",
  },
] as const;

export const RESULT_ROWS: Array<{
  key: keyof WindComparisonTs["deltas"];
  label: string;
  unit: string;
}> = [
  { key: "carryM", label: "Carry Distance", unit: "m" },
  { key: "maxHeightM", label: "Apex Height", unit: "m" },
  { key: "flightTimeS", label: "Flight Time", unit: "s" },
  { key: "landingAngleDeg", label: "Landing Angle", unit: "°" },
  { key: "lateralM", label: "Lateral Landing Offset", unit: "m" },
];

export const FLIGHT_FIELDS = [
  { key: "launchAngleDeg", label: "Launch Angle", unit: "deg", guidance: "fxLaunchAngle" },
  { key: "launchDirectionDeg", label: "Launch Direction", unit: "deg", guidance: "fxLaunchDirection" },
  { key: "spinRpm", label: "Total Spin", unit: "rpm", guidance: "fxSpinRpm" },
  { key: "spinAxisTiltDeg", label: "Spin-Axis Tilt", unit: "deg", guidance: "fxSpinAxisTilt" },
] as const;
