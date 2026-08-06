/**
 * Web data catalog for the investigative plotting suite (epic #4120 V1).
 *
 * Mirrors the Python catalog (`rate_of_closure/plotting/catalog.py`)
 * key-for-key — the parity test pins this module's key list against the
 * pytest-exported `plotcatalog.fixture.json`. Every entry carries the
 * same label / unit / category as the desktop app; entries the TS
 * physics port cannot extract yet (angular clubhead state, impact-model
 * diagnostics — P7 WASM territory) have `extractor: null` and are
 * hidden from the web plot builder while keeping the shared key space.
 */

import { solve, type ImpactScenario } from "./impact";
import { kineticsForInput, type KineticsSeriesTs } from "./kinetics";
import {
  norm,
  type SimulationInput,
  type SimulationRunTs,
} from "./simulation";

export type PlotCategory =
  | "Input"
  | "Swing Sample"
  | "Kinetics"
  | "Impact"
  | "Launch"
  | "Flight"
  | "Metric";

export type PlotScale = "linear" | "log";

/** Everything a web extractor may read. */
export interface PlotContext {
  scenario: ImpactScenario;
  input: SimulationInput;
  run: SimulationRunTs;
}

export interface PlotVariable {
  key: string;
  label: string;
  unit: string;
  category: PlotCategory;
  scale: PlotScale;
  /** null = not extractable by the web physics port yet. */
  extractor: ((ctx: PlotContext) => number | number[]) | null;
}

const deg = (r: number): number => (r * 180.0) / Math.PI;

const impactVelocity = (ctx: PlotContext): [number, number, number] => {
  if (ctx.run.impactTimeS === null) {
    return [Number.NaN, Number.NaN, Number.NaN];
  }
  const { swing } = ctx.run;
  let best = swing[0];
  let bestDt = Infinity;
  for (const sample of swing) {
    const dt = Math.abs(sample.t - ctx.run.impactTimeS);
    if (dt < bestDt) {
      bestDt = dt;
      best = sample;
    }
  }
  return best.velocity;
};

const clubPathDeg = (ctx: PlotContext): number => {
  const v = impactVelocity(ctx);
  return deg(Math.atan2(v[2], v[0]));
};

const launchValue = (
  ctx: PlotContext,
  key: keyof NonNullable<SimulationRunTs["launch"]>,
): number => ctx.run.launch?.[key] ?? Number.NaN;

const v = (
  key: string,
  label: string,
  unit: string,
  category: PlotCategory,
  extractor: PlotVariable["extractor"],
  scale: PlotScale = "linear",
): PlotVariable => ({ key, label, unit, category, scale, extractor });

/** Kinetics entry: picks one series, NaN-filled when unsupported. */
const kv = (
  key: string,
  label: string,
  unit: string,
  pick: (k: KineticsSeriesTs) => number[],
): PlotVariable =>
  v(key, label, unit, "Kinetics", (c) => {
    const series = kineticsForInput(c.input);
    if (!series) return c.run.swing.map(() => Number.NaN);
    return pick(series);
  });

/** The catalog, in the pinned display order. */
export const PLOT_CATALOG: PlotVariable[] = [
  v("input.clubhead_speed_mph", "Clubhead Speed", "mph", "Input",
    (c) => c.scenario.clubheadSpeedMph),
  v("input.omega_plane_dps", "In-Plane Rotation (SPV)", "deg/s", "Input",
    (c) => c.scenario.omegaPlaneDps),
  v("input.omega_shaft_dps", "About-Shaft Rotation (HTV)", "deg/s", "Input",
    (c) => c.scenario.omegaShaftDps),
  v("input.lie_angle_deg", "Shaft Lie at Impact", "deg", "Input",
    (c) => c.scenario.lieAngleDeg),
  v("input.com_to_face_mm", "GC to Face Center", "mm", "Input",
    (c) => c.scenario.comToFaceMm),
  v("input.impact_offset_toe_mm", "Impact Toward Toe", "mm", "Input",
    (c) => c.scenario.impactOffsetToeMm),
  v("input.impact_offset_high_mm", "Impact Above Center", "mm", "Input",
    (c) => c.scenario.impactOffsetHighMm),
  v("input.contact_duration_us", "Contact Duration", "µs", "Input",
    (c) => c.scenario.contactDurationUs),
  v("input.plane_yaw_deg", "Plane Yaw", "deg", "Input",
    (c) => c.input.planeYawDeg),
  v("input.plane_side_tilt_deg", "Plane Side Tilt", "deg", "Input",
    (c) => c.input.planeSideTiltDeg),
  v("input.plane_forward_tilt_deg", "Plane Forward Tilt", "deg", "Input",
    (c) => c.input.planeForwardTiltDeg),
  v("input.impact_time_s", "Impact Time (τ)", "s", "Input",
    (c) => c.run.impactTimeS ?? Number.NaN),
  v("swing.time_s", "Swing Time", "s", "Swing Sample",
    (c) => c.run.swing.map((s) => s.t)),
  v("swing.x_m", "Clubhead X (Target Line)", "m", "Swing Sample",
    (c) => c.run.swing.map((s) => s.position[0])),
  v("swing.y_m", "Clubhead Y (Up)", "m", "Swing Sample",
    (c) => c.run.swing.map((s) => s.position[1])),
  v("swing.z_m", "Clubhead Z (Right)", "m", "Swing Sample",
    (c) => c.run.swing.map((s) => s.position[2])),
  v("swing.speed_mps", "Clubhead Speed", "m/s", "Swing Sample",
    (c) => c.run.swing.map((s) => norm(s.velocity))),
  // Angular clubhead state is not sampled by the TS port (P7 WASM).
  v("swing.angular_speed_dps", "Clubhead Angular Speed", "deg/s",
    "Swing Sample", null),
  // Kinetics (#4125 H2): joint torques / powers / reaction forces from
  // the TS inverse-dynamics mirror; NaN for sources without joint
  // states (manual), matching the Python catalog.
  kv("kinetics.shoulder_torque_nm", "Shoulder Net Torque", "N·m",
    (k) => k.shoulderTorqueNm),
  kv("kinetics.wrist_torque_nm", "Wrist Net Torque", "N·m",
    (k) => k.wristTorqueNm),
  kv("kinetics.shoulder_gravity_torque_nm", "Shoulder Gravity Torque", "N·m",
    (k) => k.shoulderGravityTorqueNm),
  kv("kinetics.wrist_gravity_torque_nm", "Wrist Gravity Torque", "N·m",
    (k) => k.wristGravityTorqueNm),
  kv("kinetics.shoulder_damping_torque_nm", "Shoulder Damping Torque", "N·m",
    (k) => k.shoulderDampingTorqueNm),
  kv("kinetics.wrist_damping_torque_nm", "Wrist Damping Torque", "N·m",
    (k) => k.wristDampingTorqueNm),
  kv("kinetics.shoulder_ztcf_torque_nm", "Shoulder ZTCF Inertial Torque", "N·m",
    (k) => k.shoulderZtcfTorqueNm),
  kv("kinetics.wrist_ztcf_torque_nm", "Wrist ZTCF Inertial Torque", "N·m",
    (k) => k.wristZtcfTorqueNm),
  kv("kinetics.shoulder_power_w", "Shoulder Power", "W",
    (k) => k.shoulderPowerW),
  kv("kinetics.wrist_power_w", "Wrist Power", "W", (k) => k.wristPowerW),
  kv("kinetics.shoulder_force_n", "Shoulder Reaction Force", "N",
    (k) => k.shoulderForceN),
  kv("kinetics.wrist_force_n", "Wrist Reaction Force", "N",
    (k) => k.wristForceN),
  kv("kinetics.clubhead_force_n", "Clubhead Force", "N",
    (k) => k.clubheadForceN),
  kv("kinetics.shoulder_ztcf_force_n", "Shoulder ZTCF Reaction Force", "N",
    (k) => k.shoulderZtcfForceN),
  kv("kinetics.wrist_ztcf_force_n", "Wrist ZTCF Reaction Force", "N",
    (k) => k.wristZtcfForceN),
  kv("kinetics.clubhead_ztcf_force_n", "Clubhead ZTCF Force", "N",
    (k) => k.clubheadZtcfForceN),
  v("impact.clubhead_speed_mps", "Delivered Clubhead Speed", "m/s", "Impact",
    (c) => norm(impactVelocity(c))),
  v("impact.club_path_deg", "Club Path", "deg", "Impact", clubPathDeg),
  v("impact.attack_angle_deg", "Attack Angle", "deg", "Impact", (c) => {
    const vel = impactVelocity(c);
    return deg(Math.atan2(vel[1], Math.hypot(vel[0], vel[2])));
  }),
  // Impact-model diagnostics beyond the delivery are P7 WASM territory.
  v("impact.spin_loft_deg", "Spin Loft", "deg", "Impact", null),
  // Face is delivered square (face angle 0), so face-to-path = -path.
  v("impact.face_to_path_deg", "Face to Path", "deg", "Impact",
    (c) => -clubPathDeg(c)),
  v("impact.spin_axis_tilt_deg", "Spin Axis Tilt", "deg", "Impact", null),
  v("impact.energy_transfer_j", "Impact Energy Transfer", "J", "Impact", null),
  v("launch.ball_speed_mph", "Ball Speed", "mph", "Launch",
    (c) => launchValue(c, "ballSpeedMph")),
  v("launch.launch_angle_deg", "Launch Angle", "deg", "Launch",
    (c) => launchValue(c, "launchAngleDeg")),
  v("launch.launch_azimuth_deg", "Launch Direction", "deg", "Launch",
    (c) => launchValue(c, "launchAzimuthDeg")),
  v("launch.spin_rpm", "Total Spin", "rpm", "Launch",
    (c) => launchValue(c, "spinRpm")),
  v("flight.time_s", "Flight Time", "s", "Flight",
    (c) => c.run.flight.map((p) => p.time)),
  v("flight.x_m", "Downrange Distance", "m", "Flight",
    (c) => c.run.flight.map((p) => p.position[0])),
  v("flight.y_m", "Height", "m", "Flight",
    (c) => c.run.flight.map((p) => p.position[1])),
  v("flight.z_m", "Lateral (Right of Target)", "m", "Flight",
    (c) => c.run.flight.map((p) => p.position[2])),
  v("flight.speed_mps", "Ball Speed", "m/s", "Flight",
    (c) => c.run.flight.map((p) => norm(p.velocity))),
  v("metric.carry_m", "Carry Distance", "m", "Metric",
    (c) => launchValue(c, "carryM")),
  v("metric.max_height_m", "Apex Height", "m", "Metric",
    (c) => launchValue(c, "maxHeightM")),
  v("metric.flight_time_s", "Flight Time", "s", "Metric",
    (c) => launchValue(c, "flightTimeS")),
  v("metric.landing_angle_deg", "Landing Angle", "deg", "Metric",
    (c) => launchValue(c, "landingAngleDeg")),
  v("metric.path_deviation_deg", "Impact-Point Path Deviation", "deg",
    "Metric", (c) => solve(c.scenario).pathDeviationDeg),
  v("metric.closure_rate_dps", "Closure Rate (CCV)", "deg/s", "Metric",
    (c) => solve(c.scenario).closureRateDps),
];

const BY_KEY = new Map(PLOT_CATALOG.map((entry) => [entry.key, entry]));

/** All catalog keys, in the pinned order (parity contract). */
export const catalogKeys = (): string[] =>
  PLOT_CATALOG.map((entry) => entry.key);

/** Look up one catalog entry; throws on unknown keys. */
export function catalogVariable(key: string): PlotVariable {
  const entry = BY_KEY.get(key);
  if (!entry) throw new Error(`unknown catalog key ${key}`);
  return entry;
}

/** Axis label: `Label [unit]`. */
export const axisLabel = (key: string): string => {
  const entry = catalogVariable(key);
  return entry.unit ? `${entry.label} [${entry.unit}]` : entry.label;
};

/** True when the variable yields a per-sample series. */
export const isSeries = (key: string): boolean => {
  const category = catalogVariable(key).category;
  return (
    category === "Swing Sample" ||
    category === "Kinetics" ||
    category === "Flight"
  );
};

/** Entries of one category the web port can actually extract. */
export const supportedByCategory = (
  category: PlotCategory,
): PlotVariable[] =>
  PLOT_CATALOG.filter(
    (entry) => entry.category === category && entry.extractor !== null,
  );
