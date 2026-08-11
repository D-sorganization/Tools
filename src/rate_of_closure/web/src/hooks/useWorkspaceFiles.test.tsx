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
  buildCapabilityWorkflow,
  defaultCapabilityWorkflowInputs,
} from "../model/capabilityWorkflow";
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
    capability: buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
    modules: DEFAULT_PRIMARY_VIEW_STATE,
    viewWorkspace: defaultViewWorkspace,
  };
};

interface DeferredFileRead {
  resolve: (text: string) => void;
}

const stubDeferredFileReaders = (): DeferredFileRead[] => {
  const reads: DeferredFileRead[] = [];
  vi.stubGlobal(
    "FileReader",
    class {
      result: string | null = null;
      error: Error | null = null;
      onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
      onerror: ((event: ProgressEvent<FileReader>) => void) | null = null;

      readAsText(): void {
        reads.push({
          resolve: (text: string) => {
            this.result = text;
            this.onload?.(new ProgressEvent("load") as ProgressEvent<FileReader>);
          },
        });
      }
    },
  );
  return reads;
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

  it("rejects computed capability output before applying state", async () => {
    const applySnapshot = vi.fn();
    const encoded = JSON.parse(
      createWorkspaceDocument(snapshot(), {
        documentId: "workspace.invalid.capability",
        title: "Invalid",
        appVersion: "1.14.34",
        createdAtUtc: "2026-08-11T07:00:00Z",
        modifiedAtUtc: "2026-08-11T07:00:00Z",
      }),
    );
    encoded.model_session.data.capability_request.computed_result = {};
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
      value: [new File([JSON.stringify(encoded)], "invalid-capability.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: input } as never);
    });

    await waitFor(() => expect(result.current.error).toMatch(/capability workflow/i));
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
    const reads = stubDeferredFileReaders();
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

    act(() => reads[0].resolve(encoded));

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
    delete legacy.model_session.data.capability_request;
    legacy.variation_plan = null;
    const encoded = JSON.stringify(legacy);
    const reads = stubDeferredFileReaders();
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

    act(() => reads[0].resolve(encoded));

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledTimes(1));
    expect(applySnapshot.mock.calls[0][0].variation).toEqual(latest.variation);
  });

  it("ignores an older file read that completes after a newer open", async () => {
    const first = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -700 },
    };
    const second = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -900 },
    };
    const encode = (value: ReturnType<typeof snapshot>, id: string) =>
      createWorkspaceDocument(value, {
        documentId: id,
        title: id,
        appVersion: "1.14.34",
        createdAtUtc: "2026-08-11T07:00:00Z",
        modifiedAtUtc: "2026-08-11T07:00:00Z",
      });
    const firstText = encode(first, "workspace.first");
    const secondText = encode(second, "workspace.second");
    const reads = stubDeferredFileReaders();
    const applySnapshot = vi.fn();
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const select = (text: string, name: string) => {
      const element = document.createElement("input");
      Object.defineProperty(element, "files", {
        value: [new File([text], name)],
      });
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: element } as never);
    };
    act(() => {
      select(firstText, "first.roc-workspace.json");
      select(secondText, "second.roc-workspace.json");
    });

    act(() => reads[1].resolve(secondText));
    await waitFor(() => expect(applySnapshot).toHaveBeenCalledWith(second));
    act(() => reads[0].resolve(firstText));

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledTimes(1));
  });

  it("keeps the selected file mode when another picker opens during the read", async () => {
    const opened = {
      ...snapshot(),
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -650 },
    };
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.mode.capture",
      title: "Captured Mode",
      appVersion: "1.14.34",
      createdAtUtc: "2026-08-11T07:00:00Z",
      modifiedAtUtc: "2026-08-11T07:00:00Z",
    });
    const reads = stubDeferredFileReaders();
    const applySnapshot = vi.fn();
    const applyViewWorkspace = vi.fn();
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: snapshot(),
        initialSnapshot: snapshot(),
        applySnapshot,
        applyViewWorkspace,
      }),
    );
    const element = document.createElement("input");
    Object.defineProperty(element, "files", {
      value: [new File([encoded], "captured.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: element } as never);
      result.current.handleCommand(APP_COMMAND_ID.fileImportWorkspace);
      reads[0].resolve(encoded);
    });

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledWith(opened));
    expect(applyViewWorkspace).not.toHaveBeenCalled();
  });

  it.each([
    APP_COMMAND_ID.fileNewWorkspace,
    APP_COMMAND_ID.fileCloseWorkspace,
  ])("supersedes a pending open after confirmed %s", async (command) => {
    const initial = snapshot();
    const opened = {
      ...initial,
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -620 },
    };
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.reset.supersedes",
      title: "Superseded Open",
      appVersion: "1.14.34",
      createdAtUtc: "2026-08-11T07:00:00Z",
      modifiedAtUtc: "2026-08-11T07:00:00Z",
    });
    const reads = stubDeferredFileReaders();
    const applySnapshot = vi.fn();
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: initial,
        initialSnapshot: initial,
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const element = document.createElement("input");
    Object.defineProperty(element, "files", {
      value: [new File([encoded], "pending.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: element } as never);
      result.current.handleCommand(command);
      reads[0].resolve(encoded);
    });

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledTimes(1));
    expect(applySnapshot).toHaveBeenCalledWith(initial);
  });

  it("keeps a pending open active when a dirty reset is cancelled", async () => {
    const initial = snapshot();
    const dirty = {
      ...initial,
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -500 },
    };
    const opened = {
      ...initial,
      scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -610 },
    };
    const encoded = createWorkspaceDocument(opened, {
      documentId: "workspace.cancelled.reset",
      title: "Open After Cancelled Reset",
      appVersion: "1.14.34",
      createdAtUtc: "2026-08-11T07:00:00Z",
      modifiedAtUtc: "2026-08-11T07:00:00Z",
    });
    const reads = stubDeferredFileReaders();
    const applySnapshot = vi.fn();
    vi.spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    const { result } = renderHook(() =>
      useWorkspaceFiles({
        snapshot: dirty,
        initialSnapshot: initial,
        applySnapshot,
        applyViewWorkspace: vi.fn(),
      }),
    );
    const element = document.createElement("input");
    Object.defineProperty(element, "files", {
      value: [new File([encoded], "pending.roc-workspace.json")],
    });
    act(() => {
      result.current.handleCommand(APP_COMMAND_ID.fileOpenWorkspace);
      result.current.onFileChange({ currentTarget: element } as never);
      result.current.handleCommand(APP_COMMAND_ID.fileNewWorkspace);
      reads[0].resolve(encoded);
    });

    await waitFor(() => expect(applySnapshot).toHaveBeenCalledWith(opened));
    expect(applySnapshot).toHaveBeenCalledTimes(1);
  });
});
