import fixture from "./__fixtures__/workspace_torque_parity.json";
import { describe, expect, it } from "vitest";
import { PrescribedTorqueProfile } from "./torqueProfiles";
import {
  migratedLegacyTorqueFallback,
  torqueWorkspaceDocument,
  torqueWorkspaceFromDocument,
} from "./workspaceTorqueSession";

function profiles() {
  return Object.freeze(fixture.profiles.map(PrescribedTorqueProfile.fromJsonObject));
}

describe("workspace torque-profile selection parity", () => {
  it("round-trips the Python-produced library and selection fixture", () => {
    const state = torqueWorkspaceFromDocument(fixture.selection, profiles());

    expect(torqueWorkspaceDocument(state)).toEqual(fixture.selection);
    expect(state.profiles[0].toJsonObject()).toEqual(fixture.profiles[0]);
    expect(state.runConfig.mode).toBe("prescribed");
    expect(state.runConfig.jointLocks.lockedJointIds).toEqual(["joint.wrist"]);
  });

  it.each([
    ["schema", { ...fixture.selection, schema_version: 2 }],
    ["joint", {
      ...fixture.selection,
      data: {
        ...fixture.selection.data,
        run_config: {
          ...fixture.selection.data.run_config,
          locked_joint_ids: ["joint.unknown"],
        },
      },
    }],
    ["source", {
      ...fixture.selection,
      data: {
        ...fixture.selection.data,
        selection_provenance: {
          ...fixture.selection.data.selection_provenance,
          profile_source: "drawn",
        },
      },
    }],
  ])("rejects %s corruption", (_kind, document) => {
    expect(() => torqueWorkspaceFromDocument(document, profiles())).toThrow();
  });

  it("requires an exact library match when a legacy file already carries profiles", () => {
    const state = torqueWorkspaceFromDocument(fixture.selection, profiles());
    expect(migratedLegacyTorqueFallback(state, [])).toEqual(state);
    const conflict = PrescribedTorqueProfile.fromJsonObject({
      ...fixture.profiles[0],
      profile_id: "profile.conflict.v1",
    });
    expect(() => migratedLegacyTorqueFallback(state, [conflict])).toThrow(/conflicts/i);
  });
});
