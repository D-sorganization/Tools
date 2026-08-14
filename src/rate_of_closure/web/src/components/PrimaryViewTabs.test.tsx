import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_PRIMARY_VIEW_STATE } from "../model/viewPreferences";
import { PrimaryViewTabs } from "./PrimaryViewTabs";

describe("PrimaryViewTabs", () => {
  it("announces selection and exposes a visible drag affordance", () => {
    render(
      <PrimaryViewTabs
        state={DEFAULT_PRIMARY_VIEW_STATE}
        onActiveChange={() => undefined}
        onOrderChange={() => undefined}
      />,
    );

    expect(screen.getByRole("tab", { name: "Explorer" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getAllByText("⋮⋮")).toHaveLength(
      DEFAULT_PRIMARY_VIEW_STATE.order.length,
    );
  });

  it("supports keyboard reordering without changing the active view", () => {
    const onOrderChange = vi.fn();
    render(
      <PrimaryViewTabs
        state={DEFAULT_PRIMARY_VIEW_STATE}
        onActiveChange={() => undefined}
        onOrderChange={onOrderChange}
      />,
    );

    fireEvent.keyDown(screen.getByRole("tab", { name: /Simulation/ }), {
      key: "ArrowLeft",
      altKey: true,
    });
    expect(onOrderChange).toHaveBeenCalledWith([
      "explorer",
      "simulation",
      "calculation",
      "plots",
      "flight",
      "regional-surfaces",
      "ground-playback",
      "launch-monitor-analytics",
      "capability-optimization",
      "variation",
      "putting",
      "glossary",
    ]);
  });

  it("implements arrow-key tab navigation", () => {
    const onActiveChange = vi.fn();
    render(
      <PrimaryViewTabs
        state={DEFAULT_PRIMARY_VIEW_STATE}
        onActiveChange={onActiveChange}
        onOrderChange={() => undefined}
      />,
    );

    fireEvent.keyDown(screen.getByRole("tab", { name: "Explorer" }), {
      key: "ArrowRight",
    });
    expect(onActiveChange).toHaveBeenCalledWith("calculation");
  });

  it("omits hidden modules and skips them during keyboard navigation", () => {
    const onActiveChange = vi.fn();
    render(
      <PrimaryViewTabs
        state={{
          ...DEFAULT_PRIMARY_VIEW_STATE,
          visible: DEFAULT_PRIMARY_VIEW_STATE.visible.filter(
            (id) => id !== "calculation",
          ),
        }}
        onActiveChange={onActiveChange}
        onOrderChange={() => undefined}
      />,
    );

    expect(screen.queryByRole("tab", { name: "Calculation Description" }))
      .not.toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Explorer" }), {
      key: "ArrowRight",
    });
    expect(onActiveChange).toHaveBeenCalledWith("simulation");
  });
});
