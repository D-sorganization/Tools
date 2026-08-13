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
});
