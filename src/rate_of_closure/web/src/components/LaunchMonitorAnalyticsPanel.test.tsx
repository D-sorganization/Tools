import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import * as analysis from "../model/launchMonitorAnalysis";
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

  it("links pointer and keyboard selection without recomputing retained analysis", () => {
    const { container } = render(<LaunchMonitorAnalyticsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    const fingerprint = screen.getByText(/SHA-256:/).textContent;
    const scatter = screen.getByRole("img", { name: /linked scatter plot/ });
    vi.spyOn(scatter, "getBoundingClientRect").mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 640, bottom: 250,
      width: 640, height: 250, toJSON: () => ({}),
    });

    fireEvent.pointerDown(scatter, { clientX: 52, clientY: 215 });
    expect(screen.getByText(/Retained row index 0 \(zero-based\)/))
      .toHaveTextContent(/shot demo-1.*session demo-a.*vendor TrackMan/);
    fireEvent.keyDown(scatter, { key: "End" });
    expect(screen.getByText(/Retained row index 119 \(zero-based\)/))
      .toHaveTextContent(/shot demo-120.*session demo-b.*vendor FlightScope/);
    expect(container.querySelector("circle[aria-label='Selected retained row 119']"))
      .not.toBeNull();
    expect(screen.getByText(/SHA-256:/)).toHaveTextContent(fingerprint ?? "");

    fireEvent.click(screen.getByRole("button", { name: "Load Demo" }));
    expect(screen.getByText("No retained source row selected.")).toBeInTheDocument();
  });

  it("invalidates stale results when the analysis contract changes", () => {
    render(<LaunchMonitorAnalyticsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(screen.getByRole("button", { name: "Export Analysis" })).toBeVisible();

    fireEvent.change(screen.getByTitle("Select the numeric outcome variable"), {
      target: { value: "carry_distance" },
    });

    expect(screen.queryByRole("button", { name: "Export Analysis" })).toBeNull();
    expect(screen.getByText(/Run the analysis to populate/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(screen.getByRole("button", { name: "Export Analysis" })).toBeVisible();
    fireEvent.change(screen.getByLabelText("Minimum Sample Count"), {
      target: { value: "1000000" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(screen.getByRole("alert")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Minimum Sample Count"), {
      target: { value: "10" },
    });
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("commits only the latest successful dataset replacement atomically", async () => {
    const imported = Array.from({ length: 12 }, (_, index) => ({
      x: index, y: index * index + 1,
    }));
    const reader = vi.spyOn(analysis, "readLaunchMonitorFile")
      .mockResolvedValueOnce(imported);
    render(<LaunchMonitorAnalyticsPanel />);
    const input = screen.getByLabelText("Launch monitor CSV or JSON file");

    fireEvent.change(input, { target: { files: [new File(["x,y"], "two.csv")] } });
    await waitFor(() => expect(screen.getByText(/Source: two.csv/)).toBeVisible());
    expect(screen.getByTitle("Select the numeric outcome variable")).toHaveValue("x");
    expect(screen.getByLabelText("Predictor Variables")).toHaveValue(["y"]);
    expect(screen.getByTitle("Optionally compute separate results for each group"))
      .toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(screen.getByRole("button", { name: "Export Analysis" })).toBeVisible();
    reader.mockRestore();
  });

  it("preserves current evidence on failure and ignores stale import completion", async () => {
    const reader = vi.spyOn(analysis, "readLaunchMonitorFile");
    render(<LaunchMonitorAnalyticsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    fireEvent.keyDown(screen.getByRole("img", { name: /linked scatter plot/ }), {
      key: "End",
    });
    expect(screen.getByText(/Retained row index 119/)).toBeVisible();

    reader.mockRejectedValueOnce(new Error("malformed import"));
    const input = screen.getByLabelText("Launch monitor CSV or JSON file");
    fireEvent.change(input, { target: { files: [new File(["bad"], "bad.csv")] } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("malformed import"));
    expect(screen.getByText(/Source: Built-In Demonstration Data/)).toBeVisible();
    expect(screen.getByText(/Retained row index 119/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Export Analysis" })).toBeVisible();

    let resolvePending!: (rows: analysis.LaunchMonitorRow[]) => void;
    reader.mockReturnValueOnce(new Promise((resolve) => { resolvePending = resolve; }));
    fireEvent.change(input, { target: { files: [new File(["x,y"], "late.csv")] } });
    fireEvent.click(screen.getByRole("button", { name: "Load Demo" }));
    await act(async () => resolvePending([{ x: 1, y: 2 }, { x: 2, y: 3 }, { x: 3, y: 4 }]));
    expect(screen.getByText(/Source: Built-In Demonstration Data/)).toBeVisible();
    expect(screen.getByText("No retained source row selected.")).toBeVisible();
    expect(screen.queryByRole("alert")).toBeNull();
    reader.mockRestore();
  });
});
