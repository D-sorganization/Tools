import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { APP_COMMAND_ID } from "../model/appCommands";
import {
  DEFAULT_PRIMARY_VIEW_STATE,
  type PrimaryViewState,
} from "../model/viewPreferences";
import { AppToolstrip } from "./AppToolstrip";

afterEach(cleanup);

const renderToolstrip = (
  state: PrimaryViewState = DEFAULT_PRIMARY_VIEW_STATE,
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
        onShortcutHelpOpenChange={setShortcutHelpOpen}
      />
    );
  };
  render(<Harness />);
  return { onCommand, onStateChange };
};

describe("AppToolstrip", () => {
  it("exposes responsive File, View, and Tools command surfaces", () => {
    const { onCommand } = renderToolstrip();
    const toolbar = screen.getByRole("toolbar", { name: "Application commands" });
    expect(toolbar.querySelector(".overflow-x-auto")).not.toBeNull();
    expect(toolbar.querySelector(".flex-1.min-w-0.overflow-x-auto")).not.toBeNull();
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

  it("exposes layout preset buttons and fires onLayoutPreset callback", () => {
    const onLayoutPreset = vi.fn();
    render(
      <AppToolstrip
        moduleState={DEFAULT_PRIMARY_VIEW_STATE}
        theme="dark"
        shortcutHelpOpen={false}
        onModuleStateChange={vi.fn()}
        onCommand={vi.fn()}
        onShortcutHelpOpenChange={vi.fn()}
        onLayoutPreset={onLayoutPreset}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Single" }));
    expect(onLayoutPreset).toHaveBeenCalledWith("single");
    fireEvent.click(screen.getByRole("button", { name: "Split" }));
    expect(onLayoutPreset).toHaveBeenCalledWith("split_horizontal");
    fireEvent.click(screen.getByRole("button", { name: "Grid" }));
    expect(onLayoutPreset).toHaveBeenCalledWith("grid");
  });

  it("clamps File, View, and Tools popovers within constrained viewports on toggle", () => {
    renderToolstrip();

    Object.defineProperty(document.documentElement, "clientWidth", {
      value: 520,
      configurable: true,
      writable: true,
    });

    const toolsSummary = screen.getByText("Tools");
    const toolsDetails = toolsSummary.closest("details")!;
    const toolsPopover = toolsDetails.querySelector('[aria-label="Global tools"]') as HTMLDivElement;

    vi.spyOn(toolsPopover, "getBoundingClientRect").mockReturnValue({
      left: 380,
      right: 610,
      top: 50,
      bottom: 200,
      width: 230,
      height: 150,
      x: 380,
      y: 50,
      toJSON: () => {},
    });

    toolsDetails.open = true;
    fireEvent(toolsDetails, new Event("toggle"));

    // 520 viewport with 16px gutter: maxLeft = 520 - 16 - 230 = 274.
    // 274 - 380 = -106px translation.
    expect(toolsPopover.style.transform).toBe("translateX(-106px)");

    const fileSummary = screen.getByText("File");
    const fileDetails = fileSummary.closest("details")!;
    const filePopover = fileDetails.querySelector('[aria-label="File commands"]') as HTMLDivElement;

    vi.spyOn(filePopover, "getBoundingClientRect").mockReturnValue({
      left: 320,
      right: 580,
      top: 50,
      bottom: 200,
      width: 260,
      height: 150,
      x: 320,
      y: 50,
      toJSON: () => {},
    });

    fileDetails.open = true;
    fireEvent(fileDetails, new Event("toggle"));
    // 520 - 16 - 260 = 244. 244 - 320 = -76px translation.
    expect(filePopover.style.transform).toBe("translateX(-76px)");

    const viewSummary = screen.getByText("View");
    const viewDetails = viewSummary.closest("details")!;
    const viewPopover = viewDetails.querySelector('[aria-label="Workspace modules"]') as HTMLDivElement;

    vi.spyOn(viewPopover, "getBoundingClientRect").mockReturnValue({
      left: 200,
      right: 584,
      top: 50,
      bottom: 300,
      width: 384,
      height: 250,
      x: 200,
      y: 50,
      toJSON: () => {},
    });

    viewDetails.open = true;
    fireEvent(viewDetails, new Event("toggle"));
    // 520 - 16 - 384 = 120. 120 - 200 = -80px translation.
    expect(viewPopover.style.transform).toBe("translateX(-80px)");
  });
  });
});
