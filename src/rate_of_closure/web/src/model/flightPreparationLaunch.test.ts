import { describe, expect, it } from "vitest";

import { BALL_HEIGHT_REFERENCE, DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import {
  DEFAULT_FLIGHT_EXPLORER_DRAFT,
  executionLaunchForFlightExplorerDraft,
} from "./flightPreparationLaunch";

describe("flight-explorer preparation launch", () => {
  it("maps the complete App-owned editor draft into the execution-job launch wire", () => {
    const launch = executionLaunchForFlightExplorerDraft(
      DEFAULT_FLIGHT_EXPLORER_DRAFT,
    );

    expect(launch).toMatchObject({
      frame: "flight_frame:x_forward,y_left,z_up",
      launch_angle_rad: 10.9 * Math.PI / 180,
      spin_rate_rpm: 2686,
      spin_axis_unit: [0, -1, 0],
      ball_mass_kg: 0.04593,
      ball_radius_m: 0.04267 / 2,
      air_density_kg_m3: 1.225,
      gravity_m_s2: 9.80665,
      wind_speed_m_s: 0,
      wind_direction_rad: 0,
      ball_setup: {
        support_mode: "tee",
        tee_height_m: DRIVER_TEE_HEIGHT_M,
        height_reference: BALL_HEIGHT_REFERENCE,
        ball_center_m: [0, 0.04267 / 2 + DRIVER_TEE_HEIGHT_M, 0],
      },
    });
    expect(launch.ball_speed_m_s).toBeCloseTo(167 * 0.44704, 12);
    expect(launch.azimuth_angle_rad).toBeCloseTo(0, 12);
  });

  it("maps selected steady meteorological wind and an explicit ground setup", () => {
    const launch = executionLaunchForFlightExplorerDraft({
      ...DEFAULT_FLIGHT_EXPLORER_DRAFT,
      windEnabled: true,
      windSpeedMph: 10,
      windFromDeg: 90,
      ballSetup: { supportMode: "ground", teeHeightM: 0 },
    });

    expect(launch.wind_speed_m_s).toBeCloseTo(4.4704, 12);
    expect(launch.wind_direction_rad).toBeCloseTo(-Math.PI / 2, 12);
    expect(launch.ball_setup).toMatchObject({
      support_mode: "ground",
      tee_height_m: 0,
      ball_center_m: [0, 0.04267 / 2, 0],
    });
  });

  it.each([90, 270])(
    "preserves the canonical meteorological wind-to vector at %d degrees",
    (bearingDeg) => {
      const launch = executionLaunchForFlightExplorerDraft({
        ...DEFAULT_FLIGHT_EXPLORER_DRAFT,
        windEnabled: true,
        windSpeedMph: 10,
        windFromDeg: bearingDeg,
      });
      const speed = launch.wind_speed_m_s;
      const pythonLegacyVector = [
        -speed * Math.cos(launch.wind_direction_rad),
        -speed * Math.sin(launch.wind_direction_rad),
      ];
      const bearing = bearingDeg * Math.PI / 180;
      expect(pythonLegacyVector[0]).toBeCloseTo(-speed * Math.cos(bearing), 12);
      expect(pythonLegacyVector[1]).toBeCloseTo(speed * Math.sin(bearing), 12);
    },
  );
});
