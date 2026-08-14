import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { defaultViewWorkspace, type ViewWorkspace } from "../model/viewWorkspace";
import { ViewCompositor } from "./ViewCompositor";

function Harness() {
  const [workspace, setWorkspace] = useState<ViewWorkspace>(defaultViewWorkspace);
  return (
    <ViewCompositor
      workspace={workspace}
      onWorkspaceChange={setWorkspace}
      renderViewport={(kind) => <div>{kind} host</div>}
      runIdentity="run-7"
      timeS={0.42}
    />
  );
}

describe("ViewCompositor", () => {
  it("renders real distinct hosts with one synchronized run and time", () => {
    render(<Harness />);
    fireEvent.change(screen.getByRole("combobox", { name: "Viewport layout" }), {
      target: { value: "grid" },
    });

    const hosts = ["Impact", "Swing", "Flight"].map((label) =>
      screen.getByRole("region", { name: `${label} synchronized viewport` }),
    );
    expect(new Set(hosts)).toHaveLength(3);
    for (const host of hosts) {
      expect(host).toHaveAttribute("data-run-identity", "run-7");
      expect(host).toHaveAttribute("data-playback-time", "0.420");
    }
  });

  it("keeps horizontal and grid layouts useful at desktop workbench widths", () => {
    render(<Harness />);
    const layout = screen.getByRole("combobox", { name: "Viewport layout" });

    fireEvent.change(layout, { target: { value: "split_horizontal" } });
    const impact = screen.getByRole("region", {
      name: "Impact synchronized viewport",
    });
    expect(impact.parentElement).toHaveClass("lg:grid-cols-2");

    fireEvent.change(layout, { target: { value: "grid" } });
    const flight = screen.getByRole("region", {
      name: "Flight synchronized viewport",
    });
    expect(flight).toHaveClass("lg:col-span-2");
    expect(impact).not.toHaveClass("lg:col-span-2");
  });

  it("keeps at least one known view when a user hides a viewport", () => {
    render(<Harness />);
    fireEvent.change(screen.getByRole("combobox", { name: "Viewport layout" }), {
      target: { value: "grid" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "Show Impact viewport" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Show Flight viewport" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Show Swing viewport" }));

    expect(screen.getByRole("region", { name: "Swing synchronized viewport" }))
      .toBeInTheDocument();
  });

  it("supports arrow-key tab focus and keyboard-only layout manipulation", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const impact = screen.getByRole("tab", { name: "Impact" });
    impact.focus();

    await user.keyboard("{ArrowRight}");
    const swing = screen.getByRole("tab", { name: "Swing" });
    expect(swing).toHaveFocus();
    expect(swing).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowRight}");
    const flight = screen.getByRole("tab", { name: "Flight" });
    expect(flight).toHaveFocus();
    expect(flight).toHaveAttribute("aria-selected", "true");

    await user.tab();
    const layout = screen.getByRole("combobox", { name: "Viewport layout" });
    expect(layout).toHaveFocus();
    await user.tab();
    const impactToggle = screen.getByRole("checkbox", {
      name: "Show Impact viewport",
    });
    expect(impactToggle).toHaveFocus();
    await user.keyboard(" ");
    expect(layout).toHaveValue("split_horizontal");
    expect(document.querySelectorAll("[data-viewport-id]")).toHaveLength(2);

    await user.tab();
    const swingToggle = screen.getByRole("checkbox", {
      name: "Show Swing viewport",
    });
    expect(swingToggle).toHaveFocus();
    await user.keyboard(" ");
    expect(layout).toHaveValue("grid");
    expect(document.querySelectorAll("[data-viewport-id]")).toHaveLength(3);
    await user.keyboard(" ");
    expect(layout).toHaveValue("split_horizontal");
  });
});
