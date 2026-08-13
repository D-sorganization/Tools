import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { stableGroundRegionalMaterialPlanJson } from "./model/groundRegionalPlan";
import {
  buildGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
} from "./model/regionalSurfacePlan";

describe("App regional-ground variation workspace ownership", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  afterEach(() => vi.restoreAllMocks());

  it("preserves both controlled editors across mutually exclusive navigation", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("tab", { name: "Variation" }));
    const runs = screen.getByRole("textbox", { name: "Runs" });
    fireEvent.focus(runs);
    fireEvent.change(runs, { target: { value: "37" } });
    fireEvent.blur(runs);
    fireEvent.change(screen.getByRole("combobox", { name: "Analysis execution" }), {
      target: { value: "individual" },
    });

    fireEvent.click(screen.getByRole("tab", { name: "Ground Surfaces" }));
    fireEvent.change(screen.getByLabelText("Regional plan request ID"), {
      target: { value: "edited-between-tabs" },
    });

    fireEvent.click(screen.getByRole("tab", { name: "Variation" }));
    expect(screen.getByRole("textbox", { name: "Runs" })).toHaveValue("37");
    expect(screen.getByRole("combobox", { name: "Analysis execution" }))
      .toHaveValue("individual");

    fireEvent.click(screen.getByRole("tab", { name: "Ground Surfaces" }));
    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("edited-between-tabs");
  });

  it("retains imported regional evidence after navigating away", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Ground Surfaces" }));
    const request = buildGroundRegionalSurfacePlanRequest({
      ...illustrativeRegionalSurfacePlanDraft(),
      request_id: "import-survives-navigation",
    });
    const text = stableGroundRegionalMaterialPlanJson(request);
    const file = {
      name: "regional.json",
      size: new TextEncoder().encode(text).byteLength,
      text: vi.fn().mockResolvedValue(text),
    };
    fireEvent.change(screen.getByLabelText("Import regional surface plan JSON file"), {
      target: { files: [file] },
    });
    await waitFor(() => expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("import-survives-navigation"));

    fireEvent.click(screen.getByRole("tab", { name: "Variation" }));
    fireEvent.click(screen.getByRole("tab", { name: "Ground Surfaces" }));

    expect(screen.getByLabelText("Regional plan request ID"))
      .toHaveValue("import-survives-navigation");
  });
});
