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
import { initialVariationWorkspace } from "./useVariationWorkspace";

const snapshot = () => {
  const profile = starterTorqueProfile();
  return {
    scenario: DEFAULT_SCENARIO,
    club: getClub("Driver 10.5°"),
    units: {
      speed: "mph",
      rotation: "deg/s",
      length: "mm",
      distance: "yd",
    } as const,
    simulation: {
      ballSetup: {
        supportMode: "tee" as const,
        teeHeightM: DRIVER_TEE_HEIGHT_M,
      },
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
    variation: initialVariationWorkspace(),
    modules: DEFAULT_PRIMARY_VIEW_STATE,
    viewWorkspace: defaultViewWorkspace,
  };
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("browser workspace file controller", () => {
  it("protects dirty state when New is cancelled", () => {
    const changed = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: 0 },
    };
    const applySnapshot = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: changed,
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileNewWorkspace);
    });

    expect(result.current.dirty).toBe(true);
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("parses a whole file completely before applying it", async () => {
    const applySnapshot = vi.fn();
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const invalid = new File(["{not json"], "broken.json", {
      type: "application/json",
    });
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
    const encoded = JSON.parse(
      createWorkspaceDocument(snapshot(), {
        documentId: "workspace.invalid.target",
        title: "Invalid",
        appVersion: "1.14.30",
        createdAtUtc: "2026-08-10T12:00:00Z",
        modifiedAtUtc: "2026-08-10T12:00:00Z",
      }),
    );
    encoded.model_session.data.simulation_setup.data.spatial_target.source_frame =
      "camera";
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([JSON.stringify(encoded)], "invalid-target.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() =>
      expect(result.current.error).toMatch(/source_frame|frame/i),
    );
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("rejects invalid torque selection provenance before applying state", async () => {
    const applySnapshot = vi.fn();
    const encoded = JSON.parse(
      createWorkspaceDocument(snapshot(), {
        documentId: "workspace.invalid.torque",
        title: "Invalid",
        appVersion: "1.14.32",
        createdAtUtc: "2026-08-10T12:00:00Z",
        modifiedAtUtc: "2026-08-10T12:00:00Z",
      }),
    );
    encoded.model_session.data.torque_selection.data.selection_provenance.profile_source =
      "drawn";
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
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

  it("rejects invalid variation metrics before applying state", async () => {
    const applySnapshot = vi.fn();
    const encoded = JSON.parse(
      createWorkspaceDocument(snapshot(), {
        documentId: "workspace.invalid.variation",
        title: "Invalid",
        appVersion: "1.14.34",
        createdAtUtc: "2026-08-11T07:00:00Z",
        modifiedAtUtc: "2026-08-11T07:00:00Z",
      }),
    );
    encoded.model_session.data.variation_study.data.selected_output_metrics = [
      "unknown_metric",
    ];
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([JSON.stringify(encoded)], "invalid-variation.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(result.current.error).toMatch(/metric/i));
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("applies a valid opened workspace and marks the result clean", async () => {
    const opened = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -800 },
    };
    const applySnapshot = vi.fn();
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.open.test",
      title: "Open",
      appVersion: "1.14.30",
      createdAtUtc: "2026-08-10T12:00:00Z",
      modifiedAtUtc: "2026-08-10T12:00:00Z",
    });
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
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

  it("rechecks the latest dirty state before an asynchronous open applies", async () => {
    const opened = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -800 },
    };
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.async.open",
      title: "Async Open",
      appVersion: "1.14.34",
      createdAtUtc: "2026-08-11T07:00:00Z",
      modifiedAtUtc: "2026-08-11T07:00:00Z",
    });
    let releaseRead: (() => void) | undefined;
    vi.stubGlobal(
      "FileReader",
      class {
        result: string | null = null;
        error: Error | null = null;
        onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
        onerror: ((event: ProgressEvent<FileReader>) => void) | null = null;

        readAsText(): void {
          releaseRead = () => {
            this.result = encoded;
            this.onload?.(new ProgressEvent("load") as ProgressEvent<FileReader>);
          };
        }
      },
    );
    const applySnapshot = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const initial = snapshot();
    const changed = {
      ...initial,
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: 0 },
    };
    const { result, rerender } = renderHook(
      ({ current }) =>
        useWorkspaceFiles({
          snapshot: current,
          initialSnapshot: initial,
          applySnapshot,
          applyViewWorkspace: vi.fn(),
        }),
      { initialProps: { current: initial } },
    );
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([encoded], "async.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });
    rerender({ current: changed });

    act(() => releaseRead?.());

    await waitFor(() => expect(confirm).toHaveBeenCalledTimes(1));
    expect(applySnapshot).not.toHaveBeenCalled();
  });

  it("uses the latest legacy fallback after an asynchronous read", async () => {
    const initial = snapshot();
    const latest = {
      ...initial,
      variation: {
        ...initial.variation,
        plan: { ...initial.variation.plan, seed: 99 },
      },
    };
    const legacy = JSON.parse(
      createWorkspaceDocument(initial, {
        documentId: "workspace.async.legacy",
        title: "Async Legacy Open",
        appVersion: "1.14.34",
        createdAtUtc: "2026-08-11T07:00:00Z",
        modifiedAtUtc: "2026-08-11T07:00:00Z",
      }),
    );
    legacy.model_session.schema_version = 3;
    delete legacy.model_session.data.variation_study;
    legacy.variation_plan = null;
    const encoded = JSON.stringify(legacy);
    let releaseRead: (() => void) | undefined;
    vi.stubGlobal(
      "FileReader",
      class {
        result: string | null = null;
        error: Error | null = null;
        onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
        onerror: ((event: ProgressEvent<FileReader>) => void) | null = null;

        readAsText(): void {
          releaseRead = () => {
            this.result = encoded;
            this.onload?.(new ProgressEvent("load") as ProgressEvent<FileReader>);
          };
        }
      },
    );
    const applySnapshot = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const { result, rerender } = renderHook(
      ({ current }) =>
        useWorkspaceFiles({
          snapshot: current,
          initialSnapshot: initial,
          applySnapshot,
          applyViewWorkspace: vi.fn(),
        }),
      { initialProps: { current: initial } },
    );
    const input = document.createElement("input");
    Object.defineProperty(input, "files", {
      value: [new File([encoded], "legacy.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });
    rerender({ current: latest });

    act(() => releaseRead?.());

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledTimes(1));
    expect(applySnapshot.mock.calls[0][0].variation).toEqual(latest.variation);
  });
});
