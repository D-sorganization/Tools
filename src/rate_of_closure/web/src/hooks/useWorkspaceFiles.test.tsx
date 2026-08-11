import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { APP_COMMAND_ID } from "../model/appCommands";
import { DRIVER_TEE_HEIGHT_M } from "../model/ballSetup";
import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { passiveDoublePendulumRun } from "../model/doublePendulum";
import { starterTorqueProfile } from "../model/torqueProfileEditor";
import { DEFAULT_PRIMARY_VIEW_STATE } from "../model/viewPreferences";
import { defaultViewWorkspace } from "../model/viewWorkspace";
import { createWorkspaceDocument } from "../model/workspaceSession";
import {
  boxTolerance,
  createSpatialTarget,
  targetPointFromFrame,
} from "../model/spatialTarget";
import { useWorkspaceFiles } from "./useWorkspaceFiles";

const snapshot = () => {
  const profile = starterTorqueProfile();
  return ({
  scenario: DEFAULT_SCENARIO,
  club: getClub("Driver 10.5°"),
  units: { speed: "mph", rotation: "deg/s", length: "mm", distance: "yd" } as const,
  simulation: {
    ballSetup: { supportMode: "tee" as const, teeHeightM: DRIVER_TEE_HEIGHT_M },
    ballSetupUserOverridden: false,
    spatialTarget: createSpatialTarget({
      label: "Workspace target",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([125, -4, 21], "flight"),
      tolerance: boxTolerance([4, 2, 3]),
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

afterEach(() => vi.restoreAllMocks());

describe("browser workspace file controller", () => {
  it("protects dirty state when New is cancelled", () => {
    const changed = { ...snapshot(), scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: 0 } };
    const applySnapshot = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const { result } = renderHook(() => useWorkspaceFiles({
      snapshot: changed,
      initialSnapshot: snapshot(),
      applySnapshot,
      applyViewWorkspace: vi.fn(),
    }));

    act(() => { result.current.handleCommand(APP_COMMAND_ID.fileNewWorkspace); });

    expect(result.current.dirty).toBe(true);
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("parses a whole file completely before applying it", async () => {
    const applySnapshot = vi.fn();
    const { result } = renderHook(() => useWorkspaceFiles({
      snapshot: snapshot(), initialSnapshot: snapshot(), applySnapshot,
      applyViewWorkspace: vi.fn(),
    }));
    const invalid = new File(["{not json"], "broken.json", { type: "application/json" });
    const input = document.createElement("input");
    Object.defineProperty(input, "files", { value: [invalid] });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(result.current.error).toMatch(/json/i));
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("rejects an invalid nested target before applying any workspace state", async () => {
    const applySnapshot = vi.fn();
    const encoded = JSON.parse(createWorkspaceDocument(snapshot(), {
      documentId: "workspace.invalid.target", title: "Invalid", appVersion: "1.14.30",
      createdAtUtc: "2026-08-10T12:00:00Z", modifiedAtUtc: "2026-08-10T12:00:00Z",
    }));
    encoded.model_session.data.simulation_setup.data.spatial_target.source_frame = "camera";
    const { result } = renderHook(() => useWorkspaceFiles({
      snapshot: snapshot(), initialSnapshot: snapshot(), applySnapshot,
      applyViewWorkspace: vi.fn(),
    }));
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([JSON.stringify(encoded)], "invalid-target.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(result.current.error).toMatch(/source_frame|frame/i));
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("rejects invalid torque selection provenance before applying state", async () => {
    const applySnapshot = vi.fn();
    const encoded = JSON.parse(createWorkspaceDocument(snapshot(), {
      documentId: "workspace.invalid.torque", title: "Invalid", appVersion: "1.14.32",
      createdAtUtc: "2026-08-10T12:00:00Z", modifiedAtUtc: "2026-08-10T12:00:00Z",
    }));
    encoded.model_session.data.torque_selection.data.selection_provenance
      .profile_source = "drawn";
    const { result } = renderHook(() => useWorkspaceFiles({
      snapshot: snapshot(), initialSnapshot: snapshot(), applySnapshot,
      applyViewWorkspace: vi.fn(),
    }));
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([JSON.stringify(encoded)], "invalid-torque.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(result.current.error).toMatch(/provenance/i));
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("applies a valid opened workspace and marks the result clean", async () => {
    const opened = { ...snapshot(), scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -800 } };
    const applySnapshot = vi.fn();
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.open.test", title: "Open", appVersion: "1.14.30",
      createdAtUtc: "2026-08-10T12:00:00Z", modifiedAtUtc: "2026-08-10T12:00:00Z",
    });
    const { result } = renderHook(() => useWorkspaceFiles({
      snapshot: snapshot(), initialSnapshot: snapshot(), applySnapshot,
      applyViewWorkspace: vi.fn(),
    }));
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([encoded], "opened.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledWith(opened));
    expect(result.current.error).toBeNull();
  });
});
