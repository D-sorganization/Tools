import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { stableGroundRegionalMaterialPlanJson } from "./model/groundRegionalPlan";
import {
  buildGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
} from "./model/regionalSurfacePlan";
import executionJobFixture from "./model/__fixtures__/regional_ground_execution_job_golden_v1.json";

describe("App regional-ground variation workspace ownership", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

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

  it("retains an exact imported execution job across lazy workspace navigation", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Ground Playback" }));
    const source = JSON.stringify(executionJobFixture.job);
    fireEvent.change(await screen.findByTestId(
      "regional-ground-execution-job-file-input",
    ), { target: { files: [{
      name: "persistent-job.json",
      size: new TextEncoder().encode(source).byteLength,
      arrayBuffer: vi.fn().mockResolvedValue(
        Uint8Array.from(new TextEncoder().encode(source)).buffer,
      ),
    }] } });
    expect(await screen.findByText(/Loaded persistent-job.json/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Variation" }));
    fireEvent.click(screen.getByRole("tab", { name: "Ground Playback" }));

    expect(await screen.findByText("persistent-job.json")).toBeInTheDocument();
    expect(screen.getByText(executionJobFixture.job.job_sha256)).toBeInTheDocument();
  });
});
