/**
 * Namespaced variable registry for the web variation engine (#4120 V3).
 *
 * Mirror of shared/python/swing_sim/variation/registry.py restricted to
 * the categories the browser can evaluate (delivery + launch; the swing
 * and club categories are desktop-only until the P7 WASM kernels land).
 */

import type { BallSetup } from "./ballSetup";

export const CATEGORY_DELIVERY = "swing_sim.impact.delivery";
export const CATEGORY_LAUNCH = "swing_sim.flight.launch";
export const CATEGORY_SWING = "swing_sim.swing";
export const CATEGORY_CLUB = "swing_sim.club";
export const TEE_HEIGHT_VARIATION_KEY = "swing_sim.ball_setup.tee_height_m";

export type VariationMode = "delivery" | "swing" | "launch";

export interface VariableDefTs {
  key: string;
  label: string;
  unit: string;
  default: number;
  typicalScale: number;
  guidance: string;
  applicability?: "tee_only";
}

/** Mirror of the Python registry (delivery + launch categories). */
export const VARIABLE_REGISTRY: VariableDefTs[] = [
  {
    key: `${CATEGORY_SWING}.yaw_deg`,
    label: "Swing-Plane Yaw",
    unit: "deg",
    default: 0,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg about vertical.",
  },
  {
    key: `${CATEGORY_SWING}.side_tilt_deg`,
    label: "Swing-Plane Side Tilt",
    unit: "deg",
    default: -45,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg about plane lean.",
  },
  {
    key: `${CATEGORY_SWING}.forward_tilt_deg`,
    label: "Swing-Plane Forward Tilt",
    unit: "deg",
    default: 0,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg toward or away from target.",
  },
  {
    key: `${CATEGORY_SWING}.impact_time_offset_s`,
    label: "Impact-Time Offset",
    unit: "s",
    default: 0,
    typicalScale: 0.002,
    guidance: "Typical timing jitter: 1-5 ms about peak speed.",
  },
  {
    key: `${CATEGORY_SWING}.damping_shoulder`,
    label: "Shoulder Damping",
    unit: "N·m·s",
    default: 0.4,
    typicalScale: 0.05,
    guidance: "Passive double-pendulum shoulder damping.",
  },
  {
    key: `${CATEGORY_SWING}.damping_wrist`,
    label: "Wrist Damping",
    unit: "N·m·s",
    default: 0.25,
    typicalScale: 0.05,
    guidance: "Passive double-pendulum wrist damping.",
  },
  {
    key: `${CATEGORY_CLUB}.head_mass_kg`,
    label: "Clubhead Mass",
    unit: "kg",
    default: 0.2,
    typicalScale: 0.002,
    guidance: "Manufacturing tolerance: a few grams about 200 g.",
  },
  {
    key: `${CATEGORY_CLUB}.head_moi_kg_m2`,
    label: "Clubhead MOI",
    unit: "kg·m²",
    default: 4.5e-4,
    typicalScale: 2e-5,
    guidance: "Scalar clubhead MOI about the shaft axis.",
  },
  {
    key: `${CATEGORY_CLUB}.cor`,
    label: "Coefficient of Restitution",
    unit: "",
    default: 0.83,
    typicalScale: 0.005,
    guidance: "Normal coefficient of restitution used by impact.",
  },
  {
    key: TEE_HEIGHT_VARIATION_KEY,
    label: "Tee Height",
    unit: "m",
    default: 0.0381,
    typicalScale: 0.003,
    guidance: "Ground-plane clearance to the bottom of the ball; Tee mode only.",
    applicability: "tee_only",
  },
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
    label: "Launch Direction",
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

export function keysForMode(mode: VariationMode, ballSetup?: BallSetup): string[] {
  const categories = mode === "launch"
    ? [CATEGORY_LAUNCH]
    : mode === "swing"
      ? [CATEGORY_SWING, CATEGORY_CLUB]
      : [CATEGORY_DELIVERY];
  const keys = VARIABLE_REGISTRY.filter((definition) =>
    categories.some((category) => definition.key.startsWith(category)),
  ).map((definition) => definition.key);
  if (mode === "swing") {
    keys.push(
      `${CATEGORY_DELIVERY}.impact_offset_toe_mm`,
      `${CATEGORY_DELIVERY}.impact_offset_high_mm`,
    );
  }
  if (mode !== "launch" && ballSetup?.supportMode === "tee") {
    keys.push(TEE_HEIGHT_VARIATION_KEY);
  }
  return keys;
}

export const variableLabel = (key: string): string =>
  REGISTRY_BY_KEY.get(key)?.label ?? key;

export const variableDef = (key: string): VariableDefTs | undefined =>
  REGISTRY_BY_KEY.get(key);
