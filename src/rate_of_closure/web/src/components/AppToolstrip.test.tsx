import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { APP_COMMAND_ID } from "../model/appCommands";
import goldenRequest from "../model/__fixtures__/regional_ground_variation_request_golden_v1.json";
import { regionalGroundVariationRequestFromJson } from "../model/regionalGroundVariationRequestWire";
import type { RegionalGroundVariationRequestPort } from "../model/regionalGroundVariationWorkspace";
import type { RegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import { qualifiedRegionalGroundAuthorityCapability } from "../model/regionalGroundAuthority";
import {
  DEFAULT_PRIMARY_VIEW_STATE,
  type PrimaryViewState,
} from "../model/viewPreferences";
import { AppToolstrip } from "./AppToolstrip";

afterEach(cleanup);

const renderToolstrip = (
  state: PrimaryViewState = DEFAULT_PRIMARY_VIEW_STATE,
  requestPort?: RegionalGroundVariationRequestPort,
  executionWorkspace?: RegionalGroundExecutionWorkspace,
) => {
  const onStateChange = vi.fn();
  const onCommand = vi.fn();
  const Harness = () => {
    const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
    return (
      <AppToolstrip
        moduleState={state}
        theme="dark"
        shortcutHelpOpen={shortcutHelpOpen}
        onModuleStateChange={onStateChange}
        onCommand={onCommand}
        regionalGroundVariationRequestPort={requestPort}
        regionalGroundExecutionWorkspace={executionWorkspace}
        onShortcutHelpOpenChange={setShortcutHelpOpen}
      />
    );
  };
  render(<Harness />);
  return { onCommand, onStateChange };
};

const requestPort = (): RegionalGroundVariationRequestPort => ({
  snapshot: vi.fn(() => regionalGroundVariationRequestFromJson(JSON.stringify(goldenRequest))),
  apply: vi.fn(),
});

const executionWorkspace = (): RegionalGroundExecutionWorkspace => {
  const capability = qualifiedRegionalGroundAuthorityCapability();
  return {
    authority: {
      capability, checking: false,
      controls: { submitEnabled: true, statusEnabled: false, cancelEnabled: false, resultEnabled: false },
    },
    execution: {
      phase: "idle", job: null, status: null, progress: null, failure: null, result: null, error: null,
      controls: { submitEnabled: true, statusEnabled: false, cancelEnabled: false, resultEnabled: false },
      submit: vi.fn(), cancel: vi.fn(), reconcile: vi.fn(), reset: vi.fn(),
    },
    acceptedJob: null, sourceName: null, confirmed: false,
    importFile: vi.fn(), preparationAvailable: false, preparedJobStale: false,
    prepareCurrentJob: vi.fn(),
    setConfirmed: vi.fn(), clear: vi.fn(), run: vi.fn(),
  };
};

const fileBytes = (text: string): ArrayBuffer =>
  Uint8Array.from(new TextEncoder().encode(text)).buffer;

describe("AppToolstrip", () => {
  it("exposes responsive File, View, and Tools command surfaces", () => {
    const { onCommand } = renderToolstrip();
    const toolbar = screen.getByRole("toolbar", { name: "Application commands" });
    expect(toolbar.querySelector(".overflow-x-auto")).not.toBeNull();
    expect(screen.getByText("File")).toBeInTheDocument();
    expect(screen.getByText("View")).toBeInTheDocument();
    expect(screen.getByText("Tools")).toBeInTheDocument();
    fireEvent.click(screen.getByText("View"));
    expect(onCommand).toHaveBeenCalledWith(APP_COMMAND_ID.viewManageModules);

    fireEvent.click(screen.getByText("File"));
    expect(screen.getByRole("button", { name: "New Workspace" })).toBeDisabled();
    expect(screen.getByText(/workspace document adapter/i)).toBeInTheDocument();
  });

  it("protects required modules and hides an active optional module with fallback", () => {
    const state = { ...DEFAULT_PRIMARY_VIEW_STATE, active: "plots" as const };
    const { onStateChange } = renderToolstrip(state);
    fireEvent.click(screen.getByText("View"));
    expect(screen.getByRole("checkbox", { name: "Explorer Required" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Plots" }));
    expect(onStateChange).toHaveBeenCalledWith(expect.objectContaining({
      active: "explorer",
      visible: expect.not.arrayContaining(["plots"]),
    }));
  });

  it("supports module reordering and restoring defaults", () => {
    const { onStateChange } = renderToolstrip();
    fireEvent.click(screen.getByText("View"));
    fireEvent.click(screen.getByRole("button", { name: "Move Simulation up" }));
    expect(onStateChange).toHaveBeenCalledWith(expect.objectContaining({
      order: expect.arrayContaining(["simulation"]),
    }));
    const reordered = onStateChange.mock.calls[0][0] as PrimaryViewState;
    expect(reordered.order.indexOf("simulation"))
      .toBe(DEFAULT_PRIMARY_VIEW_STATE.order.indexOf("simulation") - 1);

    fireEvent.click(screen.getByRole("button", { name: "Restore default modules" }));
    expect(onStateChange).toHaveBeenLastCalledWith(DEFAULT_PRIMARY_VIEW_STATE);
  });

  it("makes Glossary, Theme, and shortcut help first-class commands", () => {
    const { onCommand } = renderToolstrip();
    fireEvent.click(screen.getByText("Tools"));
    fireEvent.click(screen.getByRole("button", { name: "Open Glossary" }));
    expect(onCommand).toHaveBeenCalledWith(APP_COMMAND_ID.globalOpenGlossary);

    fireEvent.click(screen.getByRole("button", { name: /Toggle Theme/i }));
    expect(onCommand).toHaveBeenCalledWith(APP_COMMAND_ID.globalToggleTheme);

    fireEvent.click(screen.getByRole("button", { name: "Keyboard Shortcuts" }));
    expect(screen.getByRole("dialog", { name: "Keyboard Shortcuts" }))
      .toHaveTextContent("Alt+G");
    fireEvent.click(screen.getByRole("button", { name: "Close Keyboard Shortcuts" }));
    expect(screen.queryByRole("dialog", { name: "Keyboard Shortcuts" }))
      .not.toBeInTheDocument();
  });

  it("exposes direct Impact, Swing, and Flight view commands", () => {
    const { onCommand } = renderToolstrip();
    fireEvent.click(screen.getByRole("button", { name: "Impact" }));
    fireEvent.click(screen.getByRole("button", { name: "Swing" }));
    fireEvent.click(screen.getByRole("button", { name: "Flight" }));
    expect(onCommand.mock.calls.map(([id]) => id)).toEqual([
      APP_COMMAND_ID.viewShowImpact,
      APP_COMMAND_ID.viewShowSwing,
      APP_COMMAND_ID.viewShowFlight,
    ]);
  });

  it.each(["variation", "regional-surfaces"] as const)(
    "exposes contextual combined request controls in %s",
    (active) => {
      const port = requestPort();
      renderToolstrip({ ...DEFAULT_PRIMARY_VIEW_STATE, active }, port);

      fireEvent.click(screen.getByText("File"));
      expect(screen.getByRole("button", {
        name: "Open Regional-Ground Variation Request",
      })).toBeEnabled();
      expect(screen.getByRole("button", {
        name: "Save Regional-Ground Variation Request As",
      })).toBeEnabled();
    },
  );

  it("keeps combined request controls out of unrelated modules", () => {
    renderToolstrip(DEFAULT_PRIMARY_VIEW_STATE, requestPort());
    fireEvent.click(screen.getByText("File"));

    expect(screen.queryByRole("button", {
      name: "Open Regional-Ground Variation Request",
    })).not.toBeInTheDocument();
  });

  it("shows execution file commands only in Ground Playback with truthful availability", () => {
    const workspace = executionWorkspace();
    renderToolstrip(
      { ...DEFAULT_PRIMARY_VIEW_STATE, active: "ground-playback" },
      undefined,
      workspace,
    );
    fireEvent.click(screen.getByText("File"));

    expect(screen.getByRole("button", {
      name: "Open Regional-Ground Execution Job",
    })).toBeEnabled();
    expect(screen.queryByLabelText("Open Regional-Ground Execution Job file"))
      .not.toBeInTheDocument();
    expect(screen.getByTestId("regional-ground-execution-job-menu-file-input"))
      .toHaveAttribute("hidden");
    expect(screen.getByRole("button", {
      name: "Save Regional-Ground Execution Job As",
    })).toBeDisabled();
    expect(screen.getByRole("button", {
      name: "Save Regional-Ground Execution Result As",
    })).toBeDisabled();
    expect(screen.queryByRole("button", {
      name: "Open Regional-Ground Variation Request",
    })).not.toBeInTheDocument();
  });

  it("imports only a completely valid selected request and treats no file as cancel", async () => {
    const port = requestPort();
    const { onCommand } = renderToolstrip(
      { ...DEFAULT_PRIMARY_VIEW_STATE, active: "variation" }, port,
    );
    fireEvent.click(screen.getByText("File"));
    fireEvent.click(screen.getByRole("button", {
      name: "Open Regional-Ground Variation Request",
    }));
    expect(onCommand).toHaveBeenCalledWith(
      APP_COMMAND_ID.fileOpenRegionalGroundVariationRequest,
    );
    const input = screen.getByLabelText("Open Regional-Ground Variation Request file");

    fireEvent.change(input, { target: { files: [] } });
    expect(port.apply).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    const duplicate = JSON.stringify(goldenRequest)
      .replace('"max_rows":8', '"max_rows":8,"max_rows":9');
    fireEvent.change(input, { target: { files: [{
      name: "invalid.json", size: duplicate.length,
      arrayBuffer: vi.fn().mockResolvedValue(fileBytes(duplicate)),
    }] } });
    expect(await screen.findByRole("alert")).toHaveTextContent(/duplicate/i);
    expect(port.apply).not.toHaveBeenCalled();

    const text = JSON.stringify(goldenRequest);
    fireEvent.change(input, { target: { files: [{
      name: "valid.json", size: text.length,
      arrayBuffer: vi.fn().mockResolvedValue(fileBytes(text)),
    }] } });
    expect(await screen.findByRole("status")).toHaveTextContent(/imported valid.json/i);
    expect(port.apply).toHaveBeenCalledOnce();
  });

  it("downloads canonical bytes with truthful browser semantics", () => {
    const createUrl = vi.fn(() => "blob:combined-request");
    const revokeUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const port = requestPort();
    renderToolstrip({ ...DEFAULT_PRIMARY_VIEW_STATE, active: "regional-surfaces" }, port);
    fireEvent.click(screen.getByText("File"));

    fireEvent.click(screen.getByRole("button", {
      name: "Save Regional-Ground Variation Request As",
    }));

    expect(port.snapshot).toHaveBeenCalledOnce();
    expect(createUrl).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:combined-request");
    expect(screen.getByRole("status")).toHaveTextContent(
      /browser controls the destination and overwrite behavior/i,
    );
  });

  it("shows snapshot failures and does not start a browser download", () => {
    const createUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    const port: RegionalGroundVariationRequestPort = {
      snapshot: vi.fn(() => { throw new RangeError("explicit regional evidence required"); }),
      apply: vi.fn(),
    };
    renderToolstrip({ ...DEFAULT_PRIMARY_VIEW_STATE, active: "variation" }, port);
    fireEvent.click(screen.getByText("File"));

    fireEvent.click(screen.getByRole("button", {
      name: "Save Regional-Ground Variation Request As",
    }));

    expect(screen.getByRole("alert")).toHaveTextContent(/explicit regional evidence required/i);
    expect(createUrl).not.toHaveBeenCalled();
  });
});
