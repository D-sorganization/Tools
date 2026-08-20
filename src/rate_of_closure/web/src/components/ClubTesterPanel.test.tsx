import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ClubTesterPanel } from "./ClubTesterPanel";

describe("ClubTesterPanel component", () => {
  it("renders baseline club, counterfactual controls, and initial comparison table", () => {
    render(<ClubTesterPanel />);

    expect(screen.getByRole("region", { name: "Club tester setup" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Club tester results" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Baseline Club Preset" })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Head Mass Scale" })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Loft Delta" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Club Tester Outcome Comparison Table" })).toBeInTheDocument();
  });

  it("updates counterfactual calculations on Run Evaluation click", () => {
    render(<ClubTesterPanel />);

    const loftInput = screen.getByRole("spinbutton", { name: "Loft Delta" });
    fireEvent.change(loftInput, { target: { value: "2.0" } });

    const runBtn = screen.getByRole("button", { name: "Run Club Tester Evaluation" });
    fireEvent.click(runBtn);

    expect(screen.getByText("Evaluation completed for Driver (10.5°).")).toBeInTheDocument();
  });

  it("triggers onExportReport callback when export button is clicked", () => {
    const onExport = vi.fn();
    render(<ClubTesterPanel onExportReport={onExport} />);

    const exportBtn = screen.getByRole("button", { name: "Export Fitting Report JSON" });
    fireEvent.click(exportBtn);

    expect(onExport).toHaveBeenCalledTimes(1);
    expect(onExport.mock.calls[0][0]).toContain("golf_club.fitting_report/1");
  });

  it("toggles heavy hit coupling section", () => {
    render(<ClubTesterPanel />);

    const couplingCheckbox = screen.getByRole("checkbox", { name: "Enable Heavy Hit Coupling" });
    expect(screen.getByText(/Decoupling Fraction:/i)).toBeInTheDocument();

    fireEvent.click(couplingCheckbox);
    const runBtn = screen.getByRole("button", { name: "Run Club Tester Evaluation" });
    fireEvent.click(runBtn);

    expect(screen.queryByText(/Decoupling Fraction:/i)).not.toBeInTheDocument();
  });
});
