/** App-owned Flight Explorer draft and its exact execution-job launch adapter. */

import {
  DRIVER_TEE_HEIGHT_M,
  GOLF_BALL_RADIUS_M,
  ballSetupToJson,
  type BallSetup,
} from "./ballSetup";
import { directLaunch, type DirectLaunchInput } from "./flightExplorer";
import {
  AIR_DENSITY_KG_M3,
  GOLF_BALL_MASS_KG,
  GRAVITY_M_S2,
} from "./impactPhysics";
import type { LaunchDirectionConvention } from "./launchDirection";
import {
  parseRegionalGroundExecutionLaunch,
  type ExecutionJobLaunch,
} from "./regionalGroundExecutionJob";

export type FlightExplorerSpeedUnit = "mph" | "m/s";

export interface FlightExplorerDraft {
  readonly speed: number;
  readonly speedUnit: FlightExplorerSpeedUnit;
  readonly directionConvention: LaunchDirectionConvention;
  readonly launchAngleDeg: number;
  readonly launchDirectionDeg: number;
  readonly spinRpm: number;
  readonly spinAxisTiltDeg: number;
  readonly windEnabled: boolean;
  readonly windSpeedMph: number;
  readonly windFromDeg: number;
  readonly ballSetup: BallSetup;
  readonly ballSetupUserOverridden: boolean;
}

export const FLIGHT_EXPLORER_SPEED_UNITS: Readonly<
  Record<FlightExplorerSpeedUnit, number>
> = Object.freeze({ mph: 1, "m/s": 2.236936292054402 });

export const DEFAULT_FLIGHT_EXPLORER_DRAFT: FlightExplorerDraft = Object.freeze({
  speed: 167,
  speedUnit: "mph",
  directionConvention: "app_native",
  launchAngleDeg: 10.9,
  launchDirectionDeg: 0,
  spinRpm: 2686,
  spinAxisTiltDeg: 0,
  windEnabled: false,
  windSpeedMph: 10,
  windFromDeg: 0,
  ballSetup: Object.freeze({
    supportMode: "tee",
    teeHeightM: DRIVER_TEE_HEIGHT_M,
  }),
  ballSetupUserOverridden: false,
});

/** Build the existing direct-launch input; no flight or ground physics runs here. */
export const directLaunchInputForFlightExplorerDraft = (
  draft: FlightExplorerDraft,
): DirectLaunchInput => ({
  ballSpeedMph: draft.speed * FLIGHT_EXPLORER_SPEED_UNITS[draft.speedUnit],
  launchAngleDeg: draft.launchAngleDeg,
  launchDirectionDeg: draft.launchDirectionDeg,
  launchDirectionConvention: draft.directionConvention,
  spinRpm: draft.spinRpm,
  spinAxisTiltDeg: draft.spinAxisTiltDeg,
});

/** Map the complete current editor snapshot into the strict Python request launch wire. */
export const executionLaunchForFlightExplorerDraft = (
  draft: FlightExplorerDraft,
): ExecutionJobLaunch => {
  const launch = directLaunch(directLaunchInputForFlightExplorerDraft(draft));
  return parseRegionalGroundExecutionLaunch({
    frame: "flight_frame:x_forward,y_left,z_up",
    ball_speed_m_s: launch.ballSpeedMps,
    launch_angle_rad: launch.launchAngleRad,
    azimuth_angle_rad: launch.azimuthRad,
    spin_rate_rpm: launch.spinRpm,
    spin_axis_unit: launch.spinAxis,
    ball_mass_kg: GOLF_BALL_MASS_KG,
    ball_radius_m: GOLF_BALL_RADIUS_M,
    air_density_kg_m3: AIR_DENSITY_KG_M3,
    gravity_m_s2: GRAVITY_M_S2,
    wind_speed_m_s: draft.windEnabled
      ? draft.windSpeedMph / FLIGHT_EXPLORER_SPEED_UNITS["m/s"] : 0,
    // Python's legacy scalar adapter resolves
    // v_wind=(-speed*cos(direction), -speed*sin(direction), 0), while the
    // canonical meteorological bearing resolves (-speed*cos(bearing),
    // +speed*sin(bearing), 0). Negating the bearing preserves the canonical
    // flight-frame wind-to vector at this strict wire boundary.
    wind_direction_rad: draft.windEnabled ? -draft.windFromDeg * Math.PI / 180 : 0,
    ball_setup: ballSetupToJson(draft.ballSetup),
  });
};
