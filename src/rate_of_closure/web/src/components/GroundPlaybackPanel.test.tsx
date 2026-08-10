import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
    expect(screen.getByText("flight-to-ground-result/v1")).toBeInTheDocument();
    expect(screen.getByText("literature-default-2026-08")).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();

    const trajectory = screen.getByRole("table", { name: "Ground trajectory evidence" });
    expect(within(trajectory).getByRole("columnheader", { name: "vx m/s" })).toBeInTheDocument();
    expect(within(trajectory).getByRole("columnheader", { name: "ωz rad/s" })).toBeInTheDocument();
    const trajectoryCells = within(within(trajectory).getAllByRole("row")[1]).getAllByRole("cell");
    expect(trajectoryCells[6]).toHaveTextContent("0.976000");
    expect(trajectoryCells[11]).toHaveTextContent("-2.810304");

    const events = screen.getByRole("table", { name: "Ground event evidence" });
    expect(within(events).getByRole("columnheader", { name: "vx before m/s" })).toBeInTheDocument();
    expect(within(events).getByRole("columnheader", { name: "ωz after rad/s" })).toBeInTheDocument();
    const eventCells = within(within(events).getAllByRole("row")[1]).getAllByRole("cell");
    expect(eventCells[5]).toHaveTextContent("1.000000");
    expect(eventCells[8]).toHaveTextContent("0.976000");
    expect(eventCells[16]).toHaveTextContent("-2.810304");
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

  it("rejects duplicate result fields without replacing the last good result", async () => {
    render(<GroundPlaybackPanel />);
    const input = screen.getByLabelText("Import strict ground result JSON");
    fireEvent.change(input, { target: { files: [new File(
      [JSON.stringify(fixture.result)], "good.json", { type: "application/json" },
    )] } });
    expect(await screen.findByText(/Loaded good.json/)).toBeInTheDocument();

    const duplicate = JSON.stringify(fixture.result).replace(
      '"request_id":"surface-run-analytic"',
      '"request_id":"surface-run-analytic","request_id":"duplicate"',
    );
    fireEvent.change(input, { target: { files: [new File(
      [duplicate], "duplicate.json", { type: "application/json" },
    )] } });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/duplicate JSON field/i));
    expect(screen.getByRole("alert")).toHaveTextContent(/last valid result remains loaded/i);
    expect(screen.getByRole("rowheader", { name: "Carry" })).toBeInTheDocument();
    expect(screen.getByText("0.245 m")).toBeInTheDocument();
  });

  it("imports one comparison atomically and exposes a complete accessible table", async () => {
    render(<GroundPlaybackPanel />);
    fireEvent.change(screen.getByLabelText("Import strict ground result JSON"), {
      target: { files: [new File([JSON.stringify(fixture.result)], "primary.json", {
        type: "application/json",
      })] },
    });
    expect(await screen.findByText(/Loaded primary.json/)).toBeInTheDocument();
    const comparison = structuredClone(fixture.result);
    comparison.request_id = "comparison-run";
    comparison.provenance.input_sha256 = "b".repeat(64);
    const comparisonInput = screen.getByLabelText("Import ground comparison result JSON");
    fireEvent.change(comparisonInput, { target: { files: [new File(
      [JSON.stringify(comparison)], "comparison.json", { type: "application/json" },
    )] } });

    expect(await screen.findByText(/Loaded comparison comparison.json/)).toBeInTheDocument();
    expect(screen.getByLabelText("Show comparison overlay")).toBeChecked();
    const table = screen.getByRole("table", { name: "Ground result comparison table" });
    expect(within(table).getAllByRole("row")).toHaveLength(15);
    expect(within(table).getByRole("columnheader", { name: "Comparison − primary" }))
      .toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Ground comparison provenance" }))
      .toHaveTextContent("comparison-run");
    expect(screen.getByRole("button", { name: "Export ground comparison JSON" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Export ground comparison CSV" })).toBeEnabled();

    fireEvent.change(comparisonInput, { target: { files: [new File(
      ['{"schema_version":"wrong"}'], "bad.json", { type: "application/json" },
    )] } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(
      /Last valid comparison remains loaded/,
    ));
    expect(screen.getByRole("table", { name: "Ground result comparison table" }))
      .toBeInTheDocument();
    expect(screen.getByText("comparison-run")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Import strict ground result JSON"), {
      target: { files: [new File([JSON.stringify(fixture.result)], "replacement.json", {
        type: "application/json",
      })] },
    });
    expect(await screen.findByText(/Loaded replacement.json/)).toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "Ground result comparison table" }))
      .not.toBeInTheDocument();
    expect(screen.queryByLabelText("Show comparison overlay")).not.toBeInTheDocument();
  });

  it("imports a workspace atomically and exposes deterministic exports", async () => {
    render(<GroundPlaybackPanel />);
    const workspace = {
      schema_version: "rate-of-closure-ground-playback-workspace/v1",
      result: fixture.result,
      playback: { time_s: 1.205, speed: 2, loop: true },
      view: { yaw_deg: -37.5, pitch_deg: 18, zoom: 1.75 },
    };
    fireEvent.change(screen.getByLabelText("Import ground playback workspace"), {
      target: { files: [new File([JSON.stringify(workspace)], "session.json", {
        type: "application/json",
      })] },
    });

    expect(await screen.findByText(/Loaded workspace session.json/)).toBeInTheDocument();
    expect(screen.getByLabelText("Ground playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop ground playback")).toBeChecked();
    expect(screen.getByRole("status", { name: "Ground playback position" }))
      .toHaveTextContent("1.2050 s");
    expect(screen.getByRole("button", { name: "Export ground result JSON" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Export ground trajectory CSV" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Export ground events CSV" })).toBeEnabled();

    const retained = screen.getByRole("rowheader", { name: "Carry" });
    const invalid = { ...workspace, playback: { ...workspace.playback, speed: 3 } };
    fireEvent.change(screen.getByLabelText("Import ground playback workspace"), {
      target: { files: [new File([JSON.stringify(invalid)], "bad-session.json", {
        type: "application/json",
      })] },
    });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/last valid playback remains loaded/i));
    expect(retained).toBeInTheDocument();
    expect(screen.getByLabelText("Ground playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop ground playback")).toBeChecked();
  });
});
