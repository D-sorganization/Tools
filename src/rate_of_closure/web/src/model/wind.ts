/** Pure, reproducible wind-to velocity fields in the flight frame. */

import type { Vec3 } from "./impactPhysics";

export const WIND_SCHEMA_VERSION = "wind-scenario/v1" as const;
const HARMONIC_COUNT = 6;
const TURBULENCE_NORMALIZER = Math.sqrt(HARMONIC_COUNT);

export interface WindGust {
  readonly startTimeS: number;
  readonly durationS: number;
  readonly peakVelocityMps: Vec3;
}

export interface WindScenario {
  readonly schemaVersion: typeof WIND_SCHEMA_VERSION;
  readonly baseVelocityMps: Vec3;
  readonly shearFractionPer10m: number;
  readonly gusts: readonly WindGust[];
  readonly turbulenceIntensityMps: number;
  readonly seed: number;
  readonly provenance: string;
}

const finite = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
};

const vector = (value: Vec3, name: string): Vec3 => {
  if (value.length !== 3 || value.some((item) => !Number.isFinite(item))) {
    throw new RangeError(`${name} must contain three finite components`);
  }
  return value;
};

export function meteorologicalWind(
  speedMps: number,
  fromBearingDeg: number,
  verticalMps = 0,
): WindScenario {
  finite(speedMps, "speedMps");
  finite(fromBearingDeg, "fromBearingDeg");
  finite(verticalMps, "verticalMps");
  if (speedMps < 0) throw new RangeError("speedMps must be nonnegative");
  const bearing = (fromBearingDeg * Math.PI) / 180;
  return Object.freeze({
    schemaVersion: WIND_SCHEMA_VERSION,
    baseVelocityMps: [
      -speedMps * Math.cos(bearing),
      speedMps * Math.sin(bearing),
      verticalMps,
    ] as Vec3,
    shearFractionPer10m: 0,
    gusts: Object.freeze([]),
    turbulenceIntensityMps: 0,
    seed: 0,
    provenance: "user_declared_meteorological",
  });
}

const unitNoise = (seed: number, axis: number, harmonic: number, timeS: number): number => {
  const phaseSource = Math.sin(
    (seed + 1) * (axis + 1) * 12.9898 + (harmonic + 1) * 78.233,
  );
  const fractional = ((phaseSource * 43758.5453) % 1 + 1) % 1;
  const phase = fractional * 2 * Math.PI;
  const coefficient = Math.sin((seed + 11) * (axis + 3) * (harmonic + 1) * 0.731);
  const frequencyHz = 0.2 + 0.27 * harmonic;
  return coefficient * Math.sin(2 * Math.PI * frequencyHz * timeS + phase);
};

function gustVelocity(gust: WindGust, timeS: number): Vec3 {
  finite(gust.startTimeS, "gust.startTimeS");
  finite(gust.durationS, "gust.durationS");
  vector(gust.peakVelocityMps, "gust.peakVelocityMps");
  if (gust.startTimeS < 0 || gust.durationS <= 0) {
    throw new RangeError("gust time and duration must be nonnegative/positive");
  }
  const elapsed = timeS - gust.startTimeS;
  if (elapsed < 0 || elapsed > gust.durationS) return [0, 0, 0];
  const envelope = Math.sin((Math.PI * elapsed) / gust.durationS) ** 2;
  return gust.peakVelocityMps.map((value) => value * envelope) as Vec3;
}

export function windVelocityAt(scenario: WindScenario, timeS: number, positionM: Vec3): Vec3 {
  if (scenario.schemaVersion !== WIND_SCHEMA_VERSION) {
    throw new RangeError(`unsupported wind schema: ${scenario.schemaVersion}`);
  }
  finite(timeS, "timeS");
  if (timeS < 0) throw new RangeError("timeS must be nonnegative");
  vector(positionM, "positionM");
  vector(scenario.baseVelocityMps, "baseVelocityMps");
  finite(scenario.shearFractionPer10m, "shearFractionPer10m");
  finite(scenario.turbulenceIntensityMps, "turbulenceIntensityMps");
  if (scenario.shearFractionPer10m < 0 || scenario.turbulenceIntensityMps < 0) {
    throw new RangeError("wind shear and turbulence must be nonnegative");
  }
  if (!Number.isInteger(scenario.seed)) throw new RangeError("seed must be an integer");
  if (!scenario.provenance.trim()) throw new RangeError("provenance must be nonempty");
  const shear = 1 + scenario.shearFractionPer10m * Math.max(0, positionM[2]) / 10;
  const result = scenario.baseVelocityMps.map((value) => value * shear) as Vec3;
  for (const gust of scenario.gusts) {
    const contribution = gustVelocity(gust, timeS);
    result.forEach((value, axis) => { result[axis] = value + contribution[axis]; });
  }
  if (scenario.turbulenceIntensityMps > 0) {
    result.forEach((value, axis) => {
      let noise = 0;
      for (let harmonic = 0; harmonic < HARMONIC_COUNT; harmonic += 1) {
        noise += unitNoise(scenario.seed, axis, harmonic, timeS);
      }
      result[axis] = value + scenario.turbulenceIntensityMps * noise / TURBULENCE_NORMALIZER;
    });
  }
  return result;
}

export const isSteadyWind = (scenario: WindScenario): boolean =>
  scenario.shearFractionPer10m === 0 &&
  scenario.gusts.length === 0 &&
  scenario.turbulenceIntensityMps === 0;
