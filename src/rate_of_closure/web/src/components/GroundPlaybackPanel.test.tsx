import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/ground_reference_pipeline_golden_v1.json";
import { GroundPlaybackPanel } from "./GroundPlaybackPanel";

describe("GroundPlaybackPanel", () => {
  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: () => () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });
  afterEach(() => vi.restoreAllMocks());

  it("starts empty with an honest execution and geometry disclosure", () => {
    render(<GroundPlaybackPanel />);
    expect(screen.getByText(/does not execute ground physics/i)).toBeInTheDocument();
    expect(screen.getByText(/does not embed surface geometry/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /run/i })).not.toBeInTheDocument();
  });

  it("imports only a strict result and shows summary, warnings, and provenance", async () => {
    render(<GroundPlaybackPanel />);
    const resultFile = new File(
      [JSON.stringify(fixture.result)], "ground-result.json", { type: "application/json" },
    );
    fireEvent.change(screen.getByLabelText("Import strict ground result JSON"), {
      target: { files: [resultFile] },
    });

    expect(await screen.findByText(/Loaded ground-result.json/)).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "Carry" })).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "Total" })).toBeInTheDocument();
    expect(screen.getByText("STATIC_PLANE_V1")).toBeInTheDocument();
    expect(screen.getByText(/tools.rate_of_closure/)).toBeInTheDocument();
  });

  it("rejects an execution envelope atomically and retains the last result", async () => {
    render(<GroundPlaybackPanel />);
    const input = screen.getByLabelText("Import strict ground result JSON");
    fireEvent.change(input, { target: { files: [new File(
      [JSON.stringify(fixture.result)], "good.json", { type: "application/json" },
    )] } });
    expect(await screen.findByText(/Loaded good.json/)).toBeInTheDocument();
    fireEvent.change(input, { target: { files: [new File(
      [JSON.stringify(fixture)], "envelope.json", { type: "application/json" },
    )] } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("envelope.json"));
    expect(screen.getByRole("rowheader", { name: "Carry" })).toBeInTheDocument();
  });
});
