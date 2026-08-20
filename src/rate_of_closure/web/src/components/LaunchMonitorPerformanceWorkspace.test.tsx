import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LaunchMonitorPerformanceWorkspace } from "./LaunchMonitorPerformanceWorkspace";

const rows = [
  { player: "p1", session: "a", order: 1, carry: 150, lateral: -10, speed: 100, before: 3.2, after: 2 },
  { player: "p1", session: "a", order: 1, carry: 155, lateral: 5, speed: 102, before: 3.1, after: 1.9 },
  { player: "p1", session: "b", order: 2, carry: 160, lateral: 8, speed: 106, before: 3, after: 1.8 },
];

describe("LaunchMonitorPerformanceWorkspace", () => {
  it("shows directional dispersion with chart units and a clearly named proxy", () => {
    render(<LaunchMonitorPerformanceWorkspace rows={rows} sourceName="test.csv" />);
    fireEvent.change(screen.getByLabelText("Dispersion carry column"), { target: { value: "carry" } });
    fireEvent.change(screen.getByLabelText("Dispersion lateral column"), { target: { value: "lateral" } });
    fireEvent.click(screen.getByRole("button", { name: /analyze dispersion/i }));
    expect(screen.getAllByText(/yards left.*yards right/i)).toHaveLength(2);
    expect(screen.getAllByText(/radial target error/i)).toHaveLength(2);
    expect(screen.getByRole("img", { name: /dispersion plot.*yards/i })).toBeInTheDocument();
  });

  it("fails trend and strokes gained closed until required trust/source inputs exist", () => {
    render(<LaunchMonitorPerformanceWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.getByRole("button", { name: /run session trend/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /calculate user-supplied sg/i })).toBeDisabled();
    expect(screen.getByText(/source-backed strokes gained unavailable/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Load verified strokes-gained baseline")).toBeInTheDocument();
    expect(screen.getByLabelText("Upstream strokes-gained authority URL")).toBeInTheDocument();
    expect(screen.getByLabelText("Before context column")).toBeInTheDocument();
    expect(screen.getByLabelText("Before target or hole column")).toBeInTheDocument();
    expect(screen.getByLabelText("Attest strokes-gained grouping identities and longitudinal order")).not.toBeChecked();
    expect(screen.getByRole("button", { name: /calculate source-backed sg/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /run longitudinal inference/i })).toBeDisabled();
  });

  it("exposes save/load analysis and plot/data export with explanatory titles", () => {
    render(<LaunchMonitorPerformanceWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.getByRole("button", { name: /save performance analysis/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Load saved performance analysis")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /export plot svg/i })).toHaveAttribute("title", expect.stringMatching(/units/i));
    expect(screen.getByRole("button", { name: /export backing data/i })).toBeInTheDocument();
  });
});
