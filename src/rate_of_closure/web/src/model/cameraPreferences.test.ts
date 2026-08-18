import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/camera_preferences_v1.json";
import { movingSubjectCameraState, withManualOrbit } from "./cameraCommands";
import {
  applyCameraPreference,
  cameraPreferencesDocument,
  cameraPreferencesFromDocument,
  defaultCameraPreferences,
  preferenceFromCameraState,
} from "./cameraPreferences";

describe("durable camera preferences", () => {
  it("round-trips the shared golden with isolated viewport values", () => {
    const parsed = cameraPreferencesFromDocument(fixture);
    expect(cameraPreferencesDocument(parsed)).toEqual(fixture);
    expect(parsed.viewports.impact.zoom).toBe(1.25);
    expect(parsed.viewports.swing.zoom).toBe(2.5);
    expect(parsed.viewports.flight.zoom).toBe(3.5);
  });

  it("uses the #4303 moving-subject defaults for v1 migration", () => {
    const defaults = defaultCameraPreferences().viewports;
    expect(defaults.impact).toMatchObject({
      zoom: 1,
      trackingEnabled: false,
      autoFitEnabled: false,
    });
    for (const viewportId of ["swing", "flight"] as const) {
      expect(defaults[viewportId]).toMatchObject({
        zoom: 2,
        trackingEnabled: true,
        autoFitEnabled: true,
      });
    }
  });

  it("rejects malformed and future documents", () => {
    expect(() => cameraPreferencesFromDocument({ ...fixture, format: "future/v9" }))
      .toThrow(/unsupported/);
    expect(() => cameraPreferencesFromDocument({ ...fixture, extra: true }))
      .toThrow(/invalid fields/);
    expect(() => cameraPreferencesFromDocument({
      ...fixture,
      viewports: {
        ...fixture.viewports,
        swing: { ...fixture.viewports.swing, zoom: 9 },
      },
    })).toThrow(/zoom/);
  });

  it("never persists a moving target or manual suspension", () => {
    const fallback = defaultCameraPreferences().viewports.swing;
    const runtime = withManualOrbit(
      { ...movingSubjectCameraState(), targetM: [14, 2, -3] },
      0.4,
      0.2,
    );
    const preference = preferenceFromCameraState(runtime, fallback);
    const document = cameraPreferencesDocument({
      viewports: { ...defaultCameraPreferences().viewports, swing: preference },
    });
    expect(JSON.stringify(document)).not.toContain("target");
    expect(JSON.stringify(document)).not.toContain("suspended");

    const restored = applyCameraPreference(
      { ...runtime, targetM: [99, 8, 7] },
      preference,
    );
    expect(restored.targetM).toEqual([99, 8, 7]);
    expect(restored.trackingSuspended).toBe(false);
  });
});
