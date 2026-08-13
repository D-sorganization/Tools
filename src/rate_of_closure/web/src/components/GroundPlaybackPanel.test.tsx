import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/flight_to_ground_golden_v1.json";
import regionalFixture from "../model/__fixtures__/ground_regional_execution_golden_v1.json";
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
    expect(screen.getByText(/tools-ground-reference/)).toBeInTheDocument();
    expect(screen.getByText(/tools.rate_of_closure/)).toBeInTheDocument();
    expect(screen.getByText("flight-to-ground-result/v1")).toBeInTheDocument();
    expect(screen.getByText("literature-default-2026-08")).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();

    const trajectory = screen.getByRole("table", { name: "Ground trajectory evidence" });
    expect(within(trajectory).getByRole("columnheader", { name: "vx m/s" })).toBeInTheDocument();
    expect(within(trajectory).getByRole("columnheader", { name: "ωz rad/s" })).toBeInTheDocument();
    const trajectoryCells = within(within(trajectory).getAllByRole("row")[1]).getAllByRole("cell");
    expect(trajectoryCells[6]).toHaveTextContent("24.000000");
    expect(trajectoryCells[11]).toHaveTextContent("-2.000000");

    const events = screen.getByRole("table", { name: "Ground event evidence" });
    expect(within(events).getByRole("columnheader", { name: "vx before m/s" })).toBeInTheDocument();
    expect(within(events).getByRole("columnheader", { name: "ωz after rad/s" })).toBeInTheDocument();
    const eventCells = within(within(events).getAllByRole("row")[1]).getAllByRole("cell");
    expect(eventCells[5]).toHaveTextContent("31.000000");
    expect(eventCells[8]).toHaveTextContent("24.000000");
    expect(eventCells[16]).toHaveTextContent("-2.000000");
  });

  it("imports a validated regional execution through its explicit control", async () => {
    render(<GroundPlaybackPanel />);
    const input = screen.getByLabelText("Import strict regional ground execution JSON");
    fireEvent.change(input, { target: { files: [new File(
      [JSON.stringify(regionalFixture.representable.result)], "regional.json",
      { type: "application/json" },
    )] } });
    expect(await screen.findByText(/Loaded regional.json/)).toBeInTheDocument();
    expect(screen.getByText("surface-run-analytic")).toBeInTheDocument();
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
      '"request_id":"ground-run-001"',
      '"request_id":"ground-run-001","request_id":"duplicate"',
    );
    fireEvent.change(input, { target: { files: [new File(
      [duplicate], "duplicate.json", { type: "application/json" },
    )] } });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/duplicate JSON field/i));
    expect(screen.getByRole("alert")).toHaveTextContent(/last valid result remains loaded/i);
    expect(screen.getByRole("rowheader", { name: "Carry" })).toBeInTheDocument();
    expect(screen.getByText("228.011 m")).toBeInTheDocument();
  });
});
