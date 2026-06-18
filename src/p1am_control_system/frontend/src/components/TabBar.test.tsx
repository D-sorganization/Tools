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
});
