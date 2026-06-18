import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TabBar } from "./TabBar";
import { TABS, defaultTabVisibility } from "../lib/tabs";

describe("TabBar", () => {
  it("renders one button per visible tab from the TABS array", () => {
    render(
      <TabBar
        activeTab="trends"
        visibleTabs={defaultTabVisibility()}
        onSelect={() => {}}
      />,
    );
    expect(screen.getAllByRole("tab")).toHaveLength(TABS.length);
  });

  it("hides tabs that are toggled off", () => {
    const visibility = { ...defaultTabVisibility(), routing: false };
    render(
      <TabBar activeTab="trends" visibleTabs={visibility} onSelect={() => {}} />,
    );
    expect(screen.queryByRole("tab", { name: "Signal Routing" })).toBeNull();
  });

  it("marks the active tab and fires onSelect with its id", () => {
    const onSelect = vi.fn();
    render(
      <TabBar
        activeTab="trends"
        visibleTabs={defaultTabVisibility()}
        onSelect={onSelect}
      />,
    );
    const active = screen.getByRole("tab", { name: "Trends & Monitors" });
    expect(active).toHaveAttribute("aria-selected", "true");

    fireEvent.click(screen.getByRole("tab", { name: "Signal Routing" }));
    expect(onSelect).toHaveBeenCalledWith("routing");
  });

  it("renders tabs in the caller-provided order", () => {
    const order = ["temperature", "powerSupply", "trends"] as const;
    const rest = TABS.map((t) => t.id).filter(
      (id) => !order.includes(id as (typeof order)[number]),
    );
    render(
      <TabBar
        activeTab="trends"
        visibleTabs={defaultTabVisibility()}
        onSelect={() => {}}
        order={[...order, ...rest]}
      />,
    );
    const tabs = screen.getAllByRole("tab");
    expect(tabs[0]).toHaveTextContent("Heater Controls");
    expect(tabs[1]).toHaveTextContent("Power Supply");
  });

  it("right-click menu hides a tab via onHide", () => {
    const onHide = vi.fn();
    render(
      <TabBar
        activeTab="trends"
        visibleTabs={defaultTabVisibility()}
        onSelect={() => {}}
        onHide={onHide}
      />,
    );
    fireEvent.contextMenu(screen.getByRole("tab", { name: "Signal Routing" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Hide Tab" }));
    expect(onHide).toHaveBeenCalledWith("routing");
  });

  it("right-click Move Right reorders via onReorder", () => {
    const onReorder = vi.fn();
    const order = TABS.map((t) => t.id);
    render(
      <TabBar
        activeTab="trends"
        visibleTabs={defaultTabVisibility()}
        onSelect={() => {}}
        order={order}
        onReorder={onReorder}
      />,
    );
    // Right-click the first tab and move it one slot to the right.
    fireEvent.contextMenu(screen.getAllByRole("tab")[0]);
    fireEvent.click(screen.getByRole("menuitem", { name: "Move Right" }));
    const expected = [order[1], order[0], ...order.slice(2)];
    expect(onReorder).toHaveBeenCalledWith(expected);
  });

  it("renames the temperature tab to Heater Controls", () => {
    render(
      <TabBar
        activeTab="temperature"
        visibleTabs={defaultTabVisibility()}
        onSelect={() => {}}
      />,
    );
    expect(
      screen.getByRole("tab", { name: "Heater Controls" }),
    ).toBeInTheDocument();
  });
});
