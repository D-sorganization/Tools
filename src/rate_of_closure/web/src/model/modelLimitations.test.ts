import { describe, expect, it } from "vitest";

import {
  REFERENCE_PIPELINE_LIMITATION,
  SIMULATION_MODEL_LIMITATIONS,
} from "./modelLimitations";

describe("simulation model limitations", () => {
  it("pins the machine-readable contact and impact bases", () => {
    expect(SIMULATION_MODEL_LIMITATIONS.contact_tracking.basis)
      .toBe("tracked_reference_point");
    expect(SIMULATION_MODEL_LIMITATIONS.impact_velocity.basis)
      .toBe("clubhead_reference_translation");
  });

  it("states the forced-contact and kinematic-only impact boundary", () => {
    expect(REFERENCE_PIPELINE_LIMITATION).toMatch(/kinematic analysis/i);
    expect(REFERENCE_PIPELINE_LIMITATION).toMatch(/not shaft-induced contact-point velocity/i);
    expect(REFERENCE_PIPELINE_LIMITATION).toMatch(/forced contact aligns.*reference point/i);
  });
});
