/**
 * Unit conversion system for the pendulum golf simulator.
 *
 * DbC: All conversion functions are pure and invertible.
 * DRY: Single source of truth for all unit definitions and conversions.
 *
 * @module units
 */

// ── Quantity types ───────────────────────────────────────────────────────────

export type LengthUnit = "m" | "ft" | "in" | "cm";
export type ForceUnit = "N" | "lbf";
export type TorqueUnit = "N·m" | "ft·lbf";
export type MassUnit = "kg" | "lb";
export type SpeedUnit = "m/s" | "ft/s" | "mph";
export type AngularSpeedUnit = "rad/s" | "deg/s" | "rpm";
export type EnergyUnit = "J" | "ft·lbf";
export type AngleUnit = "rad" | "deg";
export type AccelUnit = "m/s²" | "ft/s²";
export type TimeUnit = "s" | "ms";

// ── Conversion factors to SI ────────────────────────────────────────────────

const LENGTH_TO_SI: Record<LengthUnit, number> = {
  m: 1,
  ft: 0.3048,
  in: 0.0254,
  cm: 0.01,
};

const FORCE_TO_SI: Record<ForceUnit, number> = {
  N: 1,
  lbf: 4.44822,
};

const TORQUE_TO_SI: Record<TorqueUnit, number> = {
  "N·m": 1,
  "ft·lbf": 1.35582,
};

const MASS_TO_SI: Record<MassUnit, number> = {
  kg: 1,
  lb: 0.453592,
};

const SPEED_TO_SI: Record<SpeedUnit, number> = {
  "m/s": 1,
  "ft/s": 0.3048,
  mph: 0.44704,
};

const ANGULAR_SPEED_TO_SI: Record<AngularSpeedUnit, number> = {
  "rad/s": 1,
  "deg/s": Math.PI / 180,
  rpm: (2 * Math.PI) / 60,
};

const ENERGY_TO_SI: Record<EnergyUnit, number> = {
  J: 1,
  "ft·lbf": 1.35582,
};

const ANGLE_TO_SI: Record<AngleUnit, number> = {
  rad: 1,
  deg: Math.PI / 180,
};

const ACCEL_TO_SI: Record<AccelUnit, number> = {
  "m/s²": 1,
  "ft/s²": 0.3048,
};

const TIME_TO_SI: Record<TimeUnit, number> = {
  s: 1,
  ms: 0.001,
};

// ── Generic converter (DRY) ─────────────────────────────────────────────────

/** Convert a value FROM SI to the target unit. */
function fromSI(value: number, factor: number): number {
  return value / factor;
}

/** Convert a value TO SI from the source unit. */
function toSI(value: number, factor: number): number {
  return value * factor;
}

// ── Public conversion functions (small, pure) ───────────────────────────────

export function convertLength(
  value: number,
  from: LengthUnit,
  to: LengthUnit,
): number {
  return fromSI(toSI(value, LENGTH_TO_SI[from]), LENGTH_TO_SI[to]);
}

export function convertForce(
  value: number,
  from: ForceUnit,
  to: ForceUnit,
): number {
  return fromSI(toSI(value, FORCE_TO_SI[from]), FORCE_TO_SI[to]);
}

export function convertTorque(
  value: number,
  from: TorqueUnit,
  to: TorqueUnit,
): number {
  return fromSI(toSI(value, TORQUE_TO_SI[from]), TORQUE_TO_SI[to]);
}

export function convertMass(
  value: number,
  from: MassUnit,
  to: MassUnit,
): number {
  return fromSI(toSI(value, MASS_TO_SI[from]), MASS_TO_SI[to]);
}

export function convertSpeed(
  value: number,
  from: SpeedUnit,
  to: SpeedUnit,
): number {
  return fromSI(toSI(value, SPEED_TO_SI[from]), SPEED_TO_SI[to]);
}

export function convertAngularSpeed(
  value: number,
  from: AngularSpeedUnit,
  to: AngularSpeedUnit,
): number {
  return fromSI(
    toSI(value, ANGULAR_SPEED_TO_SI[from]),
    ANGULAR_SPEED_TO_SI[to],
  );
}

export function convertEnergy(
  value: number,
  from: EnergyUnit,
  to: EnergyUnit,
): number {
  return fromSI(toSI(value, ENERGY_TO_SI[from]), ENERGY_TO_SI[to]);
}

export function convertAngle(
  value: number,
  from: AngleUnit,
  to: AngleUnit,
): number {
  return fromSI(toSI(value, ANGLE_TO_SI[from]), ANGLE_TO_SI[to]);
}

export function convertAccel(
  value: number,
  from: AccelUnit,
  to: AccelUnit,
): number {
  return fromSI(toSI(value, ACCEL_TO_SI[from]), ACCEL_TO_SI[to]);
}

export function convertTime(
  value: number,
  from: TimeUnit,
  to: TimeUnit,
): number {
  return fromSI(toSI(value, TIME_TO_SI[from]), TIME_TO_SI[to]);
}

// ── Display from SI (convenience — value is always stored in SI) ────────────

export function lengthFromSI(siValue: number, unit: LengthUnit): number {
  return fromSI(siValue, LENGTH_TO_SI[unit]);
}

export function forceFromSI(siValue: number, unit: ForceUnit): number {
  return fromSI(siValue, FORCE_TO_SI[unit]);
}

export function torqueFromSI(siValue: number, unit: TorqueUnit): number {
  return fromSI(siValue, TORQUE_TO_SI[unit]);
}

export function massFromSI(siValue: number, unit: MassUnit): number {
  return fromSI(siValue, MASS_TO_SI[unit]);
}

export function speedFromSI(siValue: number, unit: SpeedUnit): number {
  return fromSI(siValue, SPEED_TO_SI[unit]);
}

export function angularSpeedFromSI(
  siValue: number,
  unit: AngularSpeedUnit,
): number {
  return fromSI(siValue, ANGULAR_SPEED_TO_SI[unit]);
}

export function energyFromSI(siValue: number, unit: EnergyUnit): number {
  return fromSI(siValue, ENERGY_TO_SI[unit]);
}

export function angleFromSI(siValue: number, unit: AngleUnit): number {
  return fromSI(siValue, ANGLE_TO_SI[unit]);
}

// ── Unit option lists (for dropdowns) ───────────────────────────────────────

export const LENGTH_UNITS: LengthUnit[] = ["m", "ft", "in", "cm"];
export const FORCE_UNITS: ForceUnit[] = ["N", "lbf"];
export const TORQUE_UNITS: TorqueUnit[] = ["N·m", "ft·lbf"];
export const MASS_UNITS: MassUnit[] = ["kg", "lb"];
export const SPEED_UNITS: SpeedUnit[] = ["m/s", "ft/s", "mph"];
export const ANGULAR_SPEED_UNITS: AngularSpeedUnit[] = [
  "rad/s",
  "deg/s",
  "rpm",
];
export const ENERGY_UNITS: EnergyUnit[] = ["J", "ft·lbf"];
export const ANGLE_UNITS: AngleUnit[] = ["rad", "deg"];

// ── Unit preferences state ──────────────────────────────────────────────────

export interface UnitPreferences {
  length: LengthUnit;
  force: ForceUnit;
  torque: TorqueUnit;
  mass: MassUnit;
  speed: SpeedUnit;
  angularSpeed: AngularSpeedUnit;
  energy: EnergyUnit;
  angle: AngleUnit;
}

export const DEFAULT_UNITS: UnitPreferences = {
  length: "m",
  force: "N",
  torque: "N·m",
  mass: "kg",
  speed: "m/s",
  angularSpeed: "deg/s",
  energy: "J",
  angle: "deg",
};

/** Format a number with unit suffix for display. */
export function formatWithUnit(
  value: number,
  unit: string,
  decimals: number = 2,
): string {
  return `${value.toFixed(decimals)} ${unit}`;
}
