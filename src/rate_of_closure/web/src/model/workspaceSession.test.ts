import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import { DEFAULT_SCENARIO } from "./impact";
import { passiveDoublePendulumRun } from "./doublePendulum";
import { starterTorqueProfile } from "./torqueProfileEditor";
import { DEFAULT_PRIMARY_VIEW_STATE } from "./viewPreferences";
import { defaultViewWorkspace } from "./viewWorkspace";
import {
  boxTolerance,
  createSpatialTarget,
  targetPointFromFrame,
} from "./spatialTarget";
import {
  createWorkspaceDocument,
  parseWorkspaceDocument,
  type WorkspaceSessionSnapshot,
} from "./workspaceSession";

const snapshot = (): WorkspaceSessionSnapshot => {
  const profile = starterTorqueProfile();
  return ({
  scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -900 },
  club: getClub("Driver 10.5°"),
  units: { speed: "mph", rotation: "deg/s", length: "mm", distance: "yd" },
  simulation: {
    ballSetup: { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
    ballSetupUserOverridden: false,
    spatialTarget: createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([137.5, 3.25, 24.25], "flight"),
      tolerance: boxTolerance([4.5, 2.5, 3.5]),
      elevationSource: "absolute",
    }),
  },
  torque: {
    profiles: Object.freeze([profile]),
    activeProfileId: profile.profileId,
    runConfig: passiveDoublePendulumRun(),
  },
  modules: DEFAULT_PRIMARY_VIEW_STATE,
  viewWorkspace: defaultViewWorkspace,
  });
};

const metadata = {
  documentId: "workspace.web.test",
  title: "Web test",
  createdAtUtc: "2026-08-10T12:00:00Z",
  modifiedAtUtc: "2026-08-10T12:01:00Z",
  appVersion: "1.14.30",
};

describe("whole workspace session contract", () => {
  it("round trips the supported live explorer state", () => {
    const encoded = createWorkspaceDocument(snapshot(), metadata);
    expect(parseWorkspaceDocument(encoded)).toEqual(snapshot());
    expect(JSON.parse(encoded).schema_version).toBe(2);
    const session = JSON.parse(encoded).model_session;
    expect(session.schema_version).toBe(3);
    expect(session.data.simulation_setup.data.ball_setup.provenance).toEqual({
      kind: "club_default",
      club_name: "Driver 10.5°",
    });
    expect(session.data.simulation_setup.data.spatial_target).toMatchObject({
      source_frame: "flight",
      tolerance: {
        kind: "box",
        half_extents_m: { x: 4.5, elevation: 2.5, right: 3.5 },
      },
    });
    expect(session.data.torque_selection.data).toMatchObject({
      active_profile_id: "profile.web.starter_drive.v1",
      selection_provenance: {
        kind: "library_profile",
        profile_source: "direct",
      },
    });
  });

  it("rejects club-default provenance that disagrees with saved geometry", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    const setup = value.model_session.data.simulation_setup.data.ball_setup.setup;
    setup.tee_height_m = 0.05;
    setup.ball_center_m[1] = 0.05 + 0.04267 / 2;
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(/club-default ball setup/i);
  });

  it("requires an explicit fallback to migrate a v1 explorer session", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 1;
    value.model_session.data = {
      scenario: value.model_session.data.scenario,
      units: value.model_session.data.units,
    };
    const text = JSON.stringify(value);
    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit.*migration/i);
    expect(parseWorkspaceDocument(text, {
      legacySimulationFallback: snapshot().simulation,
      legacyTorqueFallback: snapshot().torque,
    }).simulation).toEqual(snapshot().simulation);
  });

  it("preserves a cross-club v1 fallback as an explicit override", () => {
    const iron: WorkspaceSessionSnapshot = {
      ...snapshot(),
      club: getClub("7-Iron"),
      simulation: {
        ...snapshot().simulation,
        ballSetup: { supportMode: "ground", teeHeightM: 0 },
      },
    };
    const value = JSON.parse(createWorkspaceDocument(iron, metadata));
    value.model_session.schema_version = 1;
    value.model_session.data = {
      scenario: value.model_session.data.scenario,
      units: value.model_session.data.units,
    };
    const migrated = parseWorkspaceDocument(JSON.stringify(value), {
      legacySimulationFallback: snapshot().simulation,
      legacyTorqueFallback: snapshot().torque,
    });
    expect(migrated.simulation.ballSetup).toEqual(snapshot().simulation.ballSetup);
    expect(migrated.simulation.ballSetupUserOverridden).toBe(true);
  });

  it("requires an explicit torque fallback to migrate a v2 session", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 2;
    delete value.model_session.data.torque_selection;
    const text = JSON.stringify(value);
    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit torque/i);
    expect(parseWorkspaceDocument(text, {
      legacyTorqueFallback: snapshot().torque,
    }).torque).toEqual(snapshot().torque);
  });

  it.each([
    ["torque_unit", "lbf*ft"],
    ["coefficient_order", "descending"],
  ])("rejects noncanonical torque profile %s", (field, invalid) => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.prescribed_torque_profiles[0][field] = invalid;
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      new RegExp(field),
    );
  });

  it("rejects unsupported domain state before returning applicable values", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.variation_plan = { schema_version: 2 };
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(/variation/i);
  });

  it("rejects corrupt module and compositor documents", () => {
    const missingModule = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    missingModule.layout.module_order = ["explorer"];
    expect(() => parseWorkspaceDocument(JSON.stringify(missingModule))).toThrow(/module/i);

    const futureView = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    futureView.layout.view_workspace.data.format = "rate_of_closure.view_workspace/9";
    expect(() => parseWorkspaceDocument(JSON.stringify(futureView))).toThrow(/format/i);
  });

  it("matches the native stable identity and strict UTC metadata boundary", () => {
    const localTime = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    localTime.metadata.created_at_utc = "2026-08-10T12:00:00-07:00";
    expect(() => parseWorkspaceDocument(JSON.stringify(localTime))).toThrow(/UTC/i);

    const unstableId = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    unstableId.metadata.document_id = "workspace id with spaces";
    expect(() => parseWorkspaceDocument(JSON.stringify(unstableId))).toThrow(/identifier/i);
  });
});
