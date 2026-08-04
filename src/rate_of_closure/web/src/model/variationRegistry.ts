/**
 * Namespaced variable registry for the web variation engine (#4120 V3).
 *
 * Mirror of shared/python/swing_sim/variation/registry.py restricted to
 * the categories the browser can evaluate (delivery + launch; the swing
 * and club categories are desktop-only until the P7 WASM kernels land).
 */

export const CATEGORY_DELIVERY = "swing_sim.impact.delivery";
export const CATEGORY_LAUNCH = "swing_sim.flight.launch";

export type VariationMode = "delivery" | "launch";

export interface VariableDefTs {
  key: string;
  label: string;
  unit: string;
  default: number;
  typicalScale: number;
  guidance: string;
}

/** Mirror of the Python registry (delivery + launch categories). */
export const VARIABLE_REGISTRY: VariableDefTs[] = [
  {
    key: `${CATEGORY_DELIVERY}.clubhead_speed_mps`,
    label: "Clubhead Speed",
    unit: "m/s",
    default: 45.0,
    typicalScale: 0.5,
    guidance: "Typical shot-to-shot variation: 0.3-1 m/s.",
  },
  {
    key: `${CATEGORY_DELIVERY}.club_path_deg`,
    label: "Club Path",
    unit: "deg",
    default: 0.0,
    typicalScale: 1.0,
    guidance: "Typical shot-to-shot variation: 0.5-2 deg.",
  },
  {
    key: `${CATEGORY_DELIVERY}.face_angle_deg`,
    label: "Face Angle",
    unit: "deg",
    default: 0.0,
    typicalScale: 1.0,
    guidance:
      "Typical shot-to-shot variation: 0.5-2 deg (the dominant start-line input).",
  },
  {
    key: `${CATEGORY_DELIVERY}.attack_angle_deg`,
    label: "Attack Angle",
    unit: "deg",
    default: 0.0,
    typicalScale: 0.8,
    guidance: "Typical shot-to-shot variation: 0.5-1.5 deg.",
  },
  {
    key: `${CATEGORY_DELIVERY}.dynamic_loft_deg`,
    label: "Dynamic Loft",
    unit: "deg",
    default: 10.5,
    typicalScale: 0.8,
    guidance: "Typical shot-to-shot variation: 0.5-1.5 deg.",
  },
  {
    key: `${CATEGORY_DELIVERY}.impact_offset_toe_mm`,
    label: "Impact Toward Toe",
    unit: "mm",
    default: 0.0,
    typicalScale: 4.0,
    guidance: "Typical strike dispersion: 3-8 mm across the face.",
  },
  {
    key: `${CATEGORY_DELIVERY}.impact_offset_high_mm`,
    label: "Impact Above Center",
    unit: "mm",
    default: 0.0,
    typicalScale: 3.0,
    guidance: "Typical strike dispersion: 2-6 mm vertically.",
  },
  {
    key: `${CATEGORY_LAUNCH}.ball_speed_mph`,
    label: "Ball Speed",
    unit: "mph",
    default: 150.0,
    typicalScale: 1.0,
    guidance: "Typical shot-to-shot variation: 0.5-2 mph.",
  },
  {
    key: `${CATEGORY_LAUNCH}.launch_angle_deg`,
    label: "Launch Angle",
    unit: "deg",
    default: 12.0,
    typicalScale: 0.5,
    guidance: "Typical shot-to-shot variation: 0.3-1 deg.",
  },
  {
    key: `${CATEGORY_LAUNCH}.launch_azimuth_deg`,
    label: "Launch Azimuth",
    unit: "deg",
    default: 0.0,
    typicalScale: 0.8,
    guidance: "Positive = right of the target line.",
  },
  {
    key: `${CATEGORY_LAUNCH}.spin_rpm`,
    label: "Total Spin",
    unit: "rpm",
    default: 2600.0,
    typicalScale: 100.0,
    guidance: "Typical shot-to-shot variation: 50-300 rpm.",
  },
  {
    key: `${CATEGORY_LAUNCH}.spin_axis_deg`,
    label: "Spin-Axis Tilt",
    unit: "deg",
    default: 0.0,
    typicalScale: 1.5,
    guidance: "Positive = fade/slice side.",
  },
];

const REGISTRY_BY_KEY = new Map(VARIABLE_REGISTRY.map((d) => [d.key, d]));

export function keysForMode(mode: VariationMode): string[] {
  const category = mode === "launch" ? CATEGORY_LAUNCH : CATEGORY_DELIVERY;
  return VARIABLE_REGISTRY.filter((d) => d.key.startsWith(category)).map(
    (d) => d.key,
  );
}

export const variableLabel = (key: string): string =>
  REGISTRY_BY_KEY.get(key)?.label ?? key;

export const variableDef = (key: string): VariableDefTs | undefined =>
  REGISTRY_BY_KEY.get(key);
