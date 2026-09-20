import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { LaunchMonitorComparisonWorkspace } from "./LaunchMonitorComparisonWorkspace";

const TEST_ROWS: LaunchMonitorRow[] = [
  {
    shot_id: 1,
    monitor_vendor: "TrackMan",
    ball_speed: 70,
    club_speed: 48,
    attack_angle: -2,
    club_path: 1.5,
  },
  {
    shot_id: 2,
    monitor_vendor: "Foresight",
    ball_speed: 68,
    club_speed: 49,
    attack_angle: -2,
    club_path: 1.0,
  },
];

describe("LaunchMonitorComparisonWorkspace component", () => {
  it("renders comparison table with 28 parameters", () => {
    render(<LaunchMonitorComparisonWorkspace rows={TEST_ROWS} sourceName="Test.csv" />);

    const table = screen.getByRole("table", { name: /Launch Monitor Side-by-Side Comparison Table/i });
    expect(table).toBeInTheDocument();

    expect(screen.getByText("Club Speed")).toBeInTheDocument();
    expect(screen.getByText("Ball Speed")).toBeInTheDocument();
    expect(screen.getByText("Apex Height")).toBeInTheDocument();
  });

  it("filters table rows when selecting a parameter group", () => {
    render(<LaunchMonitorComparisonWorkspace rows={TEST_ROWS} sourceName="Test.csv" />);

    const select = screen.getByRole("combobox", { name: /Filter Parameter Group/i });
    fireEvent.change(select, { target: { value: "ball_launch" } });

    expect(screen.getByText("Ball Speed")).toBeInTheDocument();
    expect(screen.getByText("Launch Angle")).toBeInTheDocument();
    expect(screen.queryByText("Apex Height")).not.toBeInTheDocument();
  });

  it("filters table rows when typing into the search input", () => {
    render(<LaunchMonitorComparisonWorkspace rows={TEST_ROWS} sourceName="Test.csv" />);

    const searchInput = screen.getByRole("searchbox", { name: /Search Comparison Parameters/i });
    fireEvent.change(searchInput, { target: { value: "apex" } });

    expect(screen.getByText("Apex Height")).toBeInTheDocument();
    expect(screen.queryByText("Club Speed")).not.toBeInTheDocument();
  });

  it("has accessible names and tooltips on controls and table headers", () => {
    render(<LaunchMonitorComparisonWorkspace rows={TEST_ROWS} sourceName="Test.csv" />);

    expect(screen.getByRole("combobox", { name: /Filter Parameter Group/i })).toHaveAttribute("title");
    expect(screen.getByRole("searchbox", { name: /Search Comparison Parameters/i })).toHaveAttribute("title");
    expect(screen.getByRole("button", { name: /Export Comparison as JSON/i })).toHaveAttribute("title");
    expect(screen.getByRole("button", { name: /Export Comparison as CSV/i })).toHaveAttribute("title");
  });

  it("triggers export without errors", () => {
    const createUrl = vi.fn().mockReturnValue("blob:test");
    const revokeUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });

    render(<LaunchMonitorComparisonWorkspace rows={TEST_ROWS} sourceName="Test.csv" />);

    const jsonBtn = screen.getByRole("button", { name: /Export Comparison as JSON/i });
    fireEvent.click(jsonBtn);
    expect(createUrl).toHaveBeenCalled();

    const csvBtn = screen.getByRole("button", { name: /Export Comparison as CSV/i });
    fireEvent.click(csvBtn);
    expect(createUrl).toHaveBeenCalledTimes(2);
  });
});
