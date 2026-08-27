/**
 * Namespaced variable registry for the web variation engine (#4120 V3).
 *
 * Mirror of shared/python/swing_sim/variation/registry.py restricted to
 * the categories the browser can evaluate (delivery + launch; the swing
 * and club categories are desktop-only until the P7 WASM kernels land).
 */

import type { BallSetup } from "./ballSetup";
import { capabilityFor } from "./locusExecutionCapabilities";

export const CATEGORY_DELIVERY = "swing_sim.impact.delivery";
export const CATEGORY_LAUNCH = "swing_sim.flight.launch";

// Regional-ground adapter inputs. These are extension keys registered
// through the shared seam rather than VARIABLE_REGISTRY entries, mirroring
// register_ground_variation_variables() on the Python side; the key strings
// must stay byte-identical to regional_ground_variation_dataset.py.
export const GROUND_NORMAL_RESTITUTION_KEY =
  `${CATEGORY_LAUNCH}.ground_normal_restitution`;
export const GROUND_ROLLING_RESISTANCE_KEY =
  `${CATEGORY_LAUNCH}.ground_rolling_resistance`;
export const CATEGORY_SWING = "swing_sim.swing";
export const CATEGORY_CLUB = "swing_sim.club";
export const TEE_HEIGHT_VARIATION_KEY = "swing_sim.ball_setup.tee_height_m";
export const LOCALIZED_TORQUE_DURATION_S = 1.5;

export type VariationMode = "delivery" | "swing" | "launch";

export interface VariableDefTs {
  key: string;
  label: string;
  unit: string;
  default: number;
  typicalScale: number;
  guidance: string;
  applicability?: "tee_only" | "localized_torque_only";
}

/** Mirror of the Python registry (delivery + launch categories). */
export const VARIABLE_REGISTRY: VariableDefTs[] = [
  {
    key: `${CATEGORY_SWING}.yaw_deg`,
    label: "Swing-Plane Yaw",
    unit: "deg",
    default: 0,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg about vertical. Source: 3-D motion-capture swing-plane studies collected in the AffineDrift dossier.",
  },
  {
    key: `${CATEGORY_SWING}.side_tilt_deg`,
    label: "Swing-Plane Side Tilt",
    unit: "deg",
    default: -45,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg about the plane lean. Source: 3-D motion-capture swing-plane studies collected in the AffineDrift dossier.",
  },
  {
    key: `${CATEGORY_SWING}.forward_tilt_deg`,
    label: "Swing-Plane Forward Tilt",
    unit: "deg",
    default: 0,
    typicalScale: 1.5,
    guidance: "Typical variation: 1-3 deg toward/away from target. Source: 3-D motion-capture swing-plane studies collected in the AffineDrift dossier.",
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
    guidance: "Typical variation: 0.02-0.1 N·m·s about the 0.4 golf default. Source: double-pendulum golf-swing literature parameters used by swing_sim.",
  },
  {
    key: `${CATEGORY_SWING}.damping_wrist`,
    label: "Wrist Damping",
    unit: "N·m·s",
    default: 0.25,
    typicalScale: 0.05,
    guidance: "Typical variation: 0.02-0.1 N·m·s about the 0.25 golf default. Source: double-pendulum golf-swing literature parameters used by swing_sim.",
  },
  {
    key: `${CATEGORY_SWING}.shoulder_commanded_torque_offset_nm`,
    label: "Shoulder Commanded Torque Offset",
    unit: "N·m",
    default: 0,
    typicalScale: 2,
    guidance: "Additive double-pendulum command over a required half-open time window at joint.shoulder.",
    applicability: "localized_torque_only",
  },
  {
    key: `${CATEGORY_SWING}.wrist_commanded_torque_offset_nm`,
    label: "Wrist Commanded Torque Offset",
    unit: "N·m",
    default: 0,
    typicalScale: 1,
    guidance: "Additive double-pendulum command over a required half-open time window at joint.wrist.",
    applicability: "localized_torque_only",
  },
  {
    key: `${CATEGORY_CLUB}.head_mass_kg`,
    label: "Clubhead Mass",
    unit: "kg",
    default: 0.2,
    typicalScale: 0.002,
    guidance: "Manufacturing tolerance: a few grams about 200 g. Source: shared swing_sim impact constants (driver head, USGA COR limit region).",
  },
  {
    key: `${CATEGORY_CLUB}.head_moi_kg_m2`,
    label: "Clubhead MOI",
    unit: "kg·m²",
    default: 4.5e-4,
    typicalScale: 2e-5,
    guidance: "Typical driver MOI spread about 4.5e-4 kg·m². Source: shared swing_sim impact constants (driver head, USGA COR limit region).",
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
    guidance: "Applicable only when Ball Support is Tee. Height is measured from the ground plane to the bottom of the ball.",
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
    key: `${CATEGORY_DELIVERY}.lie_deg`,
    label: "Residual Lie Rotation",
    unit: "deg",
    default: 0.0,
    typicalScale: 0.5,
    guidance: "Typical variation: within 1 deg of square. Source: AffineDrift launch-monitor frame conventions.",
  },
  {
    key: `${CATEGORY_DELIVERY}.impact_offset_toe_mm`,
    label: "Impact Toward Toe",
    unit: "mm",
    default: 0.0,
    typicalScale: 4.0,
    guidance: "Typical strike dispersion: 3-8 mm across the face. Source: published robot-test impact maps.",
  },
  {
    key: `${CATEGORY_DELIVERY}.impact_offset_high_mm`,
    label: "Impact Above Center",
    unit: "mm",
    default: 0.0,
    typicalScale: 3.0,
    guidance: "Typical strike dispersion: 2-6 mm vertically. Source: published robot-test impact maps.",
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
  {
    key: GROUND_NORMAL_RESTITUTION_KEY,
    label: "Ground Normal Restitution",
    unit: "1",
    default: 0.4,
    typicalScale: 0.05,
    guidance: "Base-surface normal restitution for a regional-ground study.",
  },
  {
    key: GROUND_ROLLING_RESISTANCE_KEY,
    label: "Ground Rolling Resistance",
    unit: "1",
    default: 0.04,
    typicalScale: 0.01,
    guidance: "Base-surface rolling resistance for a regional-ground study.",
  },
];

const REGISTRY_BY_KEY = new Map(VARIABLE_REGISTRY.map((d) => [d.key, d]));

const UNIT_DIMENSIONS: Readonly<Record<string, string>> = Object.freeze({
  "": "dimensionless",
  "1": "dimensionless",
  deg: "angle",
  kg: "mass",
  "kg·m²": "moment_of_inertia",
  m: "length",
  "m/s": "speed",
  mm: "length",
  mph: "speed",
  "N·m": "torque",
  "N·m·s": "torque_time",
  rpm: "angular_frequency",
  s: "time",
});

/** Return the stable physical dimension for one registered browser unit. */
export const variableDimension = (unit: string): string => {
  const dimension = UNIT_DIMENSIONS[unit];
  if (dimension === undefined) {
    throw new Error(`registered variable unit has no dimension: ${unit}`);
  }
  return dimension;
};

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

export const localizedTorqueJointId = (
  key: string,
): "joint.shoulder" | "joint.wrist" | null => {
  const capability = capabilityFor(key);
  return capability.adapterId === "localized_joint_torque_offset/v1"
    ? capability.pointIds[0] as "joint.shoulder" | "joint.wrist"
    : null;
};
