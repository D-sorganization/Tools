import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LaunchMonitorAnalyticsPanel } from "./LaunchMonitorAnalyticsPanel";

describe("LaunchMonitorAnalyticsPanel", () => {
  it("runs the demonstration analysis and displays lineage", () => {
    render(<LaunchMonitorAnalyticsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(screen.getByText("Correlations and Multiplicity Control")).toBeInTheDocument();
    expect(screen.getByText("OLS Coefficients")).toBeInTheDocument();
    expect(screen.getByText(/SHA-256:/)).toHaveTextContent(/[a-f0-9]{64}/);
    expect(screen.getByText(/TrackMan-Comparable/, { selector: "option" })).toBeInTheDocument();
  });

  it("exposes arbitrary numeric variables and an explicit grouping selector", () => {
    render(<LaunchMonitorAnalyticsPanel />);
    expect(screen.getAllByRole("option", { name: "carry_distance" }).length).toBeGreaterThan(0);
    expect(screen.getByTitle("Optionally compute separate results for each group"))
      .toHaveValue("monitor_vendor");
    expect(screen.getByLabelText("Predictor Variables")).toHaveAttribute("multiple");
  });
});
