import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  MAX_REGIONAL_SURFACE_EDITOR_ROWS,
  buildGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
} from "../model/regionalSurfacePlan";
import { RegionalSurfacePlanPanel } from "./RegionalSurfacePlanPanel";

describe("RegionalSurfacePlanPanel", () => {
  it("matches Python provenance for the disclosed illustrative draft", () => {
    const request = buildGroundRegionalSurfacePlanRequest(
      illustrativeRegionalSurfacePlanDraft(),
    );
    expect(request.provenance.input_sha256).toBe(
      "2b3bf1b705bf86f5bf3cbe17970ddff63887410ad9f255200e5cfa31e5717db3",
    );
  });

  it("shows explicit qualification, SI units, and strict validated readback", () => {
    render(<RegionalSurfacePlanPanel />);

    expect(screen.getByRole("note", { name: "Regional surface qualification" }))
      .toHaveTextContent(/illustrative.*unvalidated/i);
    expect(screen.getByLabelText("Base domain upper coordinate (m)"))
      .toHaveValue(300);
    expect(screen.getByLabelText("Overlay 1 lower coordinate (m)"))
      .toHaveValue(120);

    fireEvent.click(screen.getByRole("button", { name: "Validate surface plan" }));

    expect(screen.getByRole("status", { name: "Regional surface plan readback" }))
      .toHaveTextContent("ground-regional-material-plan-request/v1");
    expect(screen.getByRole("status", { name: "Regional surface plan readback" }))
      .toHaveTextContent("SI");
  });

  it("links invalid interval fields to an accessible error without clearing input", () => {
    render(<RegionalSurfacePlanPanel />);
    const lower = screen.getByLabelText("Overlay 1 lower coordinate (m)");
    fireEvent.change(lower, { target: { value: "160" } });

    fireEvent.click(screen.getByRole("button", { name: "Validate surface plan" }));

    expect(screen.getByRole("alert")).toHaveTextContent("lower_coordinate_m");
    expect(lower).toHaveAttribute("aria-invalid", "true");
    expect(lower).toHaveAttribute("aria-describedby", "regional-surface-plan-error");
    expect(lower).toHaveValue(160);
  });

  it("keeps keyboard focus while a stable region identity is edited", () => {
    render(<RegionalSurfacePlanPanel />);
    const identity = screen.getByLabelText("Overlay 1 region ID");
    identity.focus();

    fireEvent.change(identity, { target: { value: "measured-band-a" } });

    expect(identity).toHaveValue("measured-band-a");
    expect(identity).toHaveFocus();
  });

  it("bounds overlay rows and never advertises execution or persistence", () => {
    render(<RegionalSurfacePlanPanel />);
    const add = screen.getByRole("button", { name: "Add regional overlay" });
    for (let index = 1; index < MAX_REGIONAL_SURFACE_EDITOR_ROWS; index += 1) {
      fireEvent.click(add);
    }

    expect(screen.getAllByRole("group", { name: /Regional overlay/ }))
      .toHaveLength(MAX_REGIONAL_SURFACE_EDITOR_ROWS);
    expect(add).toBeDisabled();
    expect(screen.queryByRole("button", { name: /run|play/i })).not.toBeInTheDocument();
    expect(screen.getByText(/session-only draft/i)).toBeInTheDocument();
  });
});
