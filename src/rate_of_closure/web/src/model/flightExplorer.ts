/**
 * Standalone flight-explorer logic for the web clone (epic #4120, V2).
 *
 * TypeScript twin of `rate_of_closure/simulation/flight_explorer.py`,
 * parity-banded by `flightExplorer.test.ts` against the pytest pinned
 * case: build flight-frame launch conditions from launch-monitor ball
 * numbers (app signs: launch direction and lateral + = right of target, spin
 * axis tilt + = fade side) and integrate with the Waterloo/Penner
 * model. The full 7-model picker stays Python-side until the P7 WASM
 * kernels land.
 */

import { simulateFlight, type FlightPoint, type Launch } from "./flight";
import {
  launchDirectionFromRecord,
  launchDirectionToFlightAzimuth,
  type LaunchDirectionConvention,
} from "./launchDirection";
import {
  BALL_POSITION,
  MPH_PER_MPS,
  add,
  fromFlightFrame,
  type Vec3,
} from "./simulation";
import type { WindScenario } from "./wind";

const rad = (d: number): number => (d * Math.PI) / 180.0;
const deg = (r: number): number => (r * 180.0) / Math.PI;

export interface DirectLaunchInput {
  ballSpeedMph: number;
  launchAngleDeg: number;
  /** Canonical positive-right horizontal direction. */
  launchDirectionDeg?: number;
  /** @deprecated Lossless import compatibility for pre-#4193 callers. */
  azimuthDeg?: number;
  launchDirectionConvention?: LaunchDirectionConvention;
  spinRpm: number;
  spinAxisTiltDeg: number; // + = fade side (curves right)
}

/** Twin of `launch_from_direct` (app signs -> flight frame). */
export function directLaunch(input: DirectLaunchInput): Launch {
  const values = [input.ballSpeedMph, input.launchAngleDeg, input.spinRpm, input.spinAxisTiltDeg];
  if (values.some((value) => typeof value !== "number" || !Number.isFinite(value)) ||
      input.ballSpeedMph < 1 || input.ballSpeedMph > 250 ||
      Math.abs(input.launchAngleDeg) > 89 || input.spinRpm < 0 || input.spinRpm > 15_000 ||
      Math.abs(input.spinAxisTiltDeg) > 60) {
    throw new RangeError("direct flight inputs are outside the supported domain");
  }
  const direction = launchDirectionFromRecord(input as unknown as Record<string, unknown>);
  if (!Number.isFinite(direction.degrees) || Math.abs(direction.degrees) > 45) {
    throw new RangeError("launch direction must be finite and within -45..45 degrees");
  }
  // App direction + = right; flight-frame azimuth + = left: flip. The
  // fade-side tilt (+) needs a downward (-z flight) sidespin component,
  // so the legacy spin-axis-angle decomposition gets the flipped angle
  // too (same derivation as the Python twin).
  const azimuthRad = rad(
    launchDirectionToFlightAzimuth(direction.degrees, direction.convention),
  );
  const axisAngle = -rad(input.spinAxisTiltDeg);
  const backspin = Math.cos(axisAngle);
  const sidespin = Math.sin(axisAngle);
  const spinAxis: Vec3 = [
    sidespin * Math.sin(azimuthRad),
    -backspin,
    sidespin * Math.cos(azimuthRad),
  ];
  return {
    ballSpeedMps: input.ballSpeedMph / MPH_PER_MPS,
    launchAngleRad: rad(input.launchAngleDeg),
    azimuthRad,
    spinRpm: input.spinRpm,
    spinAxis,
  };
}

export interface FlightExplorationTs {
  /** App-frame trajectory from the tee (x target, y up, z right). */
  points: FlightPoint[];
  metrics: {
    ballSpeedMph: number;
    launchAngleDeg: number;
    launchDirectionDeg: number; // + = right of target
    /** @deprecated Persistence alias retained for older exports. */
    launchAzimuthDeg: number; // + = right of target
    spinRpm: number;
    carryM: number;
    maxHeightM: number;
    flightTimeS: number;
    landingAngleDeg: number;
    lateralM: number; // + = right of target
  };
  execution: {
    readonly model: "waterloo_penner";
    readonly kernelRevision: "web-rk4-10ms-sampled-v1";
    readonly windScenario: WindScenario | null;
    readonly launch: Launch;
  };
}

export interface WindComparisonTs {
  /** Identical launch evaluated with no wind for a controlled comparison. */
  calm: FlightExplorationTs;
  /** Identical launch evaluated with the declared wind scenario. */
  wind: FlightExplorationTs;
  deltas: Pick<FlightExplorationTs["metrics"],
    "carryM" | "maxHeightM" | "flightTimeS" | "landingAngleDeg" | "lateralM">;
  scenario: WindScenario;
}

/** Twin of `explore_flight` (Waterloo/Penner only on web). */
export function exploreFlight(launch: Launch): FlightExplorationTs {
  const result = simulateFlight(launch);
  const points = result.trajectory.map((point) => ({
    ...point,
    position: add(fromFlightFrame(point.position), BALL_POSITION),
    velocity: fromFlightFrame(point.velocity),
  }));
  return {
    points,
    metrics: {
      ballSpeedMph: launch.ballSpeedMps * MPH_PER_MPS,
      launchAngleDeg: deg(launch.launchAngleRad),
      // Flight azimuth + = left; public launch direction + = right.
      launchDirectionDeg: -deg(launch.azimuthRad),
      launchAzimuthDeg: -deg(launch.azimuthRad),
      spinRpm: launch.spinRpm,
      carryM: result.carryM,
      maxHeightM: result.maxHeightM,
      flightTimeS: result.flightTimeS,
      landingAngleDeg: result.landingAngleDeg,
      // Flight lateral + = left; app lateral + = right.
      lateralM: -result.lateralM,
    },
    execution: Object.freeze({
      model: "waterloo_penner",
      kernelRevision: "web-rk4-10ms-sampled-v1",
      windScenario: launch.windScenario ?? null,
      launch: Object.freeze({
        ...launch,
        spinAxis: Object.freeze([...launch.spinAxis]) as Vec3,
        windScenario: launch.windScenario,
      }),
    }),
  };
}

/** Run common-input no-wind and wind trajectories and retain auditable deltas. */
export function compareWind(
  launch: Launch,
  scenario: WindScenario,
  execute: (request: Launch) => FlightExplorationTs = exploreFlight,
): WindComparisonTs {
  const calm = execute({ ...launch, windScenario: undefined });
  const wind = execute({ ...launch, windScenario: scenario });
  return {
    calm,
    wind,
    deltas: {
      carryM: wind.metrics.carryM - calm.metrics.carryM,
      maxHeightM: wind.metrics.maxHeightM - calm.metrics.maxHeightM,
      flightTimeS: wind.metrics.flightTimeS - calm.metrics.flightTimeS,
      landingAngleDeg: wind.metrics.landingAngleDeg - calm.metrics.landingAngleDeg,
      lateralM: wind.metrics.lateralM - calm.metrics.lateralM,
    },
    scenario,
  };
}
