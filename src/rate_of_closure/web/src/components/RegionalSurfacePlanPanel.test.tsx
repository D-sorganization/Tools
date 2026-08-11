import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MAX_REGIONAL_SURFACE_EDITOR_ROWS,
  buildGroundRegionalSurfacePlanRequest,
  editorDraftFromGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
  regionalSurfacePlanRequestForDraft,
} from "../model/regionalSurfacePlan";
import {
  MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES,
  parseGroundRegionalMaterialPlanRequest,
  stableGroundRegionalMaterialPlanJson,
} from "../model/groundRegionalPlan";
import { RegionalSurfacePlanPanel } from "./RegionalSurfacePlanPanel";

afterEach(() => vi.restoreAllMocks());

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

  it("invalidates a validated readback as soon as the draft changes", () => {
    render(<RegionalSurfacePlanPanel />);

    fireEvent.click(screen.getByRole("button", { name: "Validate surface plan" }));
    expect(screen.getByRole("status", { name: "Regional surface plan readback" }))
      .toHaveTextContent("1 overlay(s)");

    fireEvent.click(screen.getByRole("button", { name: "Add regional overlay" }));

    expect(screen.queryByRole("status", { name: "Regional surface plan readback" }))
      .not.toBeInTheDocument();
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
    expect(screen.getByText(/workspace persistence remains separate/i)).toBeInTheDocument();
  });

  it("preserves an unchanged imported editor request exactly", () => {
    const request = buildGroundRegionalSurfacePlanRequest({
      ...illustrativeRegionalSurfacePlanDraft(), request_id: "imported-plan",
    });

    const imported = editorDraftFromGroundRegionalSurfacePlanRequest(request);

    expect(regionalSurfacePlanRequestForDraft(imported, request)).toBe(request);
    expect(stableGroundRegionalMaterialPlanJson(
      regionalSurfacePlanRequestForDraft(imported, request),
    )).toBe(stableGroundRegionalMaterialPlanJson(request));

    const payload = JSON.parse(stableGroundRegionalMaterialPlanJson(request));
    payload.provenance.input_sha256 = "0".repeat(64);
    const mismatched = parseGroundRegionalMaterialPlanRequest(payload);
    expect(() => editorDraftFromGroundRegionalSurfacePlanRequest(mismatched))
      .toThrow(/digest does not match/i);
  });

  it("imports only after complete validation and retains the prior draft on error", async () => {
    render(<RegionalSurfacePlanPanel />);
    const request = buildGroundRegionalSurfacePlanRequest({
      ...illustrativeRegionalSurfacePlanDraft(), request_id: "browser-opened-plan",
    });
    const input = screen.getByLabelText("Import regional surface plan JSON file");
    const validText = stableGroundRegionalMaterialPlanJson(request);
    const valid = {
      name: "valid.json",
      size: new TextEncoder().encode(validText).byteLength,
      text: vi.fn().mockResolvedValue(validText),
    };

    fireEvent.change(input, { target: { files: [valid] } });
    await waitFor(() => expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("browser-opened-plan"));

    const corrupt = {
      name: "corrupt.json", size: 47,
      text: vi.fn().mockResolvedValue('{"request_id":"one","request_id":"two"}'),
    };
    fireEvent.change(input, { target: { files: [corrupt] } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/duplicate/i));
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("browser-opened-plan");

    const invalidSyntax = {
      name: "syntax.json", size: 9,
      text: vi.fn().mockResolvedValue("{not-json"),
    };
    fireEvent.change(input, { target: { files: [invalidSyntax] } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/invalid profile json/i));
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("browser-opened-plan");
  });

  it("rejects non-editor qualification without populating the editor", async () => {
    render(<RegionalSurfacePlanPanel />);
    const request = buildGroundRegionalSurfacePlanRequest(
      illustrativeRegionalSurfacePlanDraft(),
    );
    const payload = JSON.parse(stableGroundRegionalMaterialPlanJson(request));
    payload.provenance.producer = "external.course.authority";
    const input = screen.getByLabelText("Import regional surface plan JSON file");
    const file = {
      name: "external.json", size: 100,
      text: vi.fn().mockResolvedValue(JSON.stringify(payload)),
    };

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/editor producer/i));
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("illustrative-regional-plan");
  });

  it("rejects oversized browser files before allocating their text", async () => {
    render(<RegionalSurfacePlanPanel />);
    const input = screen.getByLabelText("Import regional surface plan JSON file");
    const text = vi.fn().mockResolvedValue("{}");
    const file = {
      name: "oversize.json", size: MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES + 1, text,
    };

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/maximum wire size/i));
    expect(text).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("illustrative-regional-plan");
  });

  it("downloads canonical bytes and revokes its object URL", () => {
    const createUrl = vi.fn(() => "blob:regional-plan");
    const revokeUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<RegionalSurfacePlanPanel />);

    fireEvent.click(screen.getByRole("button", { name: "Download regional surface plan JSON" }));

    expect(createUrl).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-plan");
  });

  it("treats an empty browser chooser as cancellation", () => {
    render(<RegionalSurfacePlanPanel />);
    const input = screen.getByLabelText("Import regional surface plan JSON file");

    fireEvent.change(input, { target: { files: [] } });

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("illustrative-regional-plan");
  });
});
