import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/ground_reference_pipeline_golden_v1.json";
import { GroundPlaybackPanel } from "./GroundPlaybackPanel";
import { downloadText } from "./variationUi";

vi.mock("./variationUi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./variationUi")>()),
  downloadText: vi.fn(),
}));

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
  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("starts empty with an honest execution and geometry disclosure", () => {
    render(<GroundPlaybackPanel />);
    expect(
      screen.getByText(/does not execute ground physics/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/does not embed surface geometry/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /run/i }),
    ).not.toBeInTheDocument();
  });

  it("imports only a strict result and shows summary, warnings, and provenance", async () => {
    render(<GroundPlaybackPanel />);
    const resultFile = new File(
      [JSON.stringify(fixture.result)],
      "ground-result.json",
      { type: "application/json" },
    );
    fireEvent.change(
      screen.getByLabelText("Import strict ground result JSON"),
      {
        target: { files: [resultFile] },
      },
    );

    expect(
      await screen.findByText(/Loaded ground-result.json/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("rowheader", { name: "Carry" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("rowheader", { name: "Total" }),
    ).toBeInTheDocument();
    expect(screen.getByText("STATIC_PLANE_V1")).toBeInTheDocument();
    expect(screen.getByText(/tools.rate_of_closure/)).toBeInTheDocument();
    expect(screen.getByText("flight-to-ground-result/v1")).toBeInTheDocument();
    expect(screen.getByText("literature-default-2026-08")).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();

    const trajectory = screen.getByRole("table", {
      name: "Ground primary trajectory evidence",
    });
    expect(
      within(trajectory).getByRole("columnheader", { name: "vx m/s" }),
    ).toBeInTheDocument();
    expect(
      within(trajectory).getByRole("columnheader", { name: "ωz rad/s" }),
    ).toBeInTheDocument();
    const trajectoryCells = within(
      within(trajectory).getAllByRole("row")[1],
    ).getAllByRole("cell");
    expect(trajectoryCells[6]).toHaveTextContent("0.976000");
    expect(trajectoryCells[11]).toHaveTextContent("-2.810304");

    const events = screen.getByRole("table", {
      name: "Ground primary event evidence",
    });
    expect(
      within(events).getByRole("columnheader", { name: "vx before m/s" }),
    ).toBeInTheDocument();
    expect(
      within(events).getByRole("columnheader", { name: "ωz after rad/s" }),
    ).toBeInTheDocument();
    const eventCells = within(
      within(events).getAllByRole("row")[1],
    ).getAllByRole("cell");
    expect(eventCells[5]).toHaveTextContent("1.000000");
    expect(eventCells[8]).toHaveTextContent("0.976000");
    expect(eventCells[16]).toHaveTextContent("-2.810304");
  });

  it("rejects an execution envelope atomically and retains the last result", async () => {
    render(<GroundPlaybackPanel />);
    const input = screen.getByLabelText("Import strict ground result JSON");
    fireEvent.change(input, {
      target: {
        files: [
          new File([JSON.stringify(fixture.result)], "good.json", {
            type: "application/json",
          }),
        ],
      },
    });
    expect(await screen.findByText(/Loaded good.json/)).toBeInTheDocument();
    fireEvent.change(input, {
      target: {
        files: [
          new File([JSON.stringify(fixture)], "envelope.json", {
            type: "application/json",
          }),
        ],
      },
    });
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("envelope.json"),
    );
    expect(
      screen.getByRole("rowheader", { name: "Carry" }),
    ).toBeInTheDocument();
  });

  it("rejects duplicate result fields without replacing the last good result", async () => {
    render(<GroundPlaybackPanel />);
    const input = screen.getByLabelText("Import strict ground result JSON");
    fireEvent.change(input, {
      target: {
        files: [
          new File([JSON.stringify(fixture.result)], "good.json", {
            type: "application/json",
          }),
        ],
      },
    });
    expect(await screen.findByText(/Loaded good.json/)).toBeInTheDocument();

    const duplicate = JSON.stringify(fixture.result).replace(
      '"request_id":"surface-run-analytic"',
      '"request_id":"surface-run-analytic","request_id":"duplicate"',
    );
    fireEvent.change(input, {
      target: {
        files: [
          new File([duplicate], "duplicate.json", { type: "application/json" }),
        ],
      },
    });

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /duplicate JSON field/i,
      ),
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      /last valid result remains loaded/i,
    );
    expect(
      screen.getByRole("rowheader", { name: "Carry" }),
    ).toBeInTheDocument();
    expect(screen.getByText("0.245 m")).toBeInTheDocument();
  });

  it("imports one comparison atomically and exposes a complete accessible table", async () => {
    render(<GroundPlaybackPanel />);
    fireEvent.change(
      screen.getByLabelText("Import strict ground result JSON"),
      {
        target: {
          files: [
            new File([JSON.stringify(fixture.result)], "primary.json", {
              type: "application/json",
            }),
          ],
        },
      },
    );
    expect(await screen.findByText(/Loaded primary.json/)).toBeInTheDocument();
    const comparison = structuredClone(fixture.result);
    comparison.request_id = "comparison-run";
    comparison.provenance.input_sha256 = "b".repeat(64);
    comparison.trajectory.forEach((point) => {
      point.time_s += 0.2;
    });
    comparison.events.forEach((event) => {
      event.time_s += 0.2;
    });
    comparison.termination.time_s += 0.2;
    const comparisonInput = screen.getByLabelText(
      "Import ground comparison result JSON",
    );
    fireEvent.change(comparisonInput, {
      target: {
        files: [
          new File([JSON.stringify(comparison)], "comparison.json", {
            type: "application/json",
          }),
        ],
      },
    });

    expect(
      await screen.findByText(/Loaded comparison comparison.json/),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Show comparison overlay")).toBeChecked();
    const table = screen.getByRole("table", {
      name: "Ground result comparison table",
    });
    expect(within(table).getAllByRole("row")).toHaveLength(15);
    expect(
      within(table).getByRole("columnheader", { name: "Comparison − primary" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Ground comparison provenance" }),
    ).toHaveTextContent("comparison-run");
    expect(
      screen.getByRole("button", { name: "Export ground comparison JSON" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Export ground comparison CSV" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", {
        name: "Export ground comparison trajectory CSV",
      }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", {
        name: "Export ground comparison events CSV",
      }),
    ).toBeEnabled();
    const comparisonTrajectory = screen.getByRole("table", {
      name: "Ground comparison trajectory evidence",
    });
    const comparisonEvents = screen.getByRole("table", {
      name: "Ground comparison event evidence",
    });
    expect(
      within(within(comparisonTrajectory).getAllByRole("row")[1]).getAllByRole(
        "cell",
      )[0],
    ).toHaveTextContent("1.205000");
    expect(
      within(within(comparisonEvents).getAllByRole("row")[1]).getAllByRole(
        "cell",
      )[1],
    ).toHaveTextContent("1.205000");

    fireEvent.click(
      screen.getByRole("button", {
        name: "Export ground comparison trajectory CSV",
      }),
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: "Export ground comparison events CSV",
      }),
    );
    const download = vi.mocked(downloadText);
    const trajectoryCall = download.mock.calls[download.mock.calls.length - 2];
    const eventCall = download.mock.calls[download.mock.calls.length - 1];
    expect(trajectoryCall?.[0]).toBe("ground-comparison-trajectory.csv");
    expect(trajectoryCall?.[1]).toMatch(/^sample_index,time_s,phase,frame/);
    expect(trajectoryCall?.[1]).toContain("\n0,1.205,impact,");
    expect(trajectoryCall?.[1]).not.toContain("\r");
    expect(trajectoryCall?.[1].endsWith("\n")).toBe(true);
    expect(eventCall?.[0]).toBe("ground-comparison-events.csv");
    expect(eventCall?.[1]).toMatch(/^sequence,event_type,time_s,frame/);
    expect(eventCall?.[1]).toContain("\n0,first_contact,1.205,");
    expect(eventCall?.[1]).not.toContain("\r");
    expect(eventCall?.[1].endsWith("\n")).toBe(true);

    fireEvent.click(screen.getByLabelText("Show comparison overlay"));
    const preservedTime = fixture.result.termination.time_s + 0.1;
    fireEvent.change(screen.getByLabelText("Ground playback absolute time"), {
      target: { value: preservedTime },
    });
    fireEvent.click(screen.getByLabelText("Show comparison overlay"));
    fireEvent.click(screen.getByLabelText("Show comparison overlay"));
    expect(screen.getByLabelText("Ground playback absolute time")).toHaveValue(
      String(preservedTime),
    );
    expect(
      screen.getByRole("table", {
        name: "Ground comparison trajectory evidence",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: "Ground comparison event evidence" }),
    ).toBeInTheDocument();

    fireEvent.change(comparisonInput, {
      target: {
        files: [
          new File(['{"schema_version":"wrong"}'], "bad.json", {
            type: "application/json",
          }),
        ],
      },
    });
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /Last valid comparison remains loaded/,
      ),
    );
    expect(
      screen.getByRole("table", { name: "Ground result comparison table" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("table", {
        name: "Ground comparison trajectory evidence",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: "Ground comparison event evidence" }),
    ).toBeInTheDocument();
    expect(screen.getByText("comparison-run")).toBeInTheDocument();

    fireEvent.change(
      screen.getByLabelText("Import strict ground result JSON"),
      {
        target: {
          files: [
            new File([JSON.stringify(fixture.result)], "replacement.json", {
              type: "application/json",
            }),
          ],
        },
      },
    );
    expect(
      await screen.findByText(/Loaded replacement.json/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("table", { name: "Ground result comparison table" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("table", {
        name: "Ground comparison trajectory evidence",
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("table", { name: "Ground comparison event evidence" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Show comparison overlay"),
    ).not.toBeInTheDocument();
  });

  it.each([
    {
      replacementLabel: "Import strict ground result JSON",
      replacementDocument: JSON.stringify(fixture.result),
      loadedPattern: /Loaded replacement.json/,
    },
    {
      replacementLabel: "Import ground playback workspace",
      replacementDocument: JSON.stringify({
        schema_version: "rate-of-closure-ground-playback-workspace/v1",
        result: fixture.result,
        playback: { time_s: 1.205, speed: 1, loop: false },
        view: { yaw_deg: -37.5, pitch_deg: 18, zoom: 1.25 },
      }),
      loadedPattern:
        /Migrated workspace replacement.json from v1.*saved as v2/i,
    },
  ])(
    "discards a comparison that finishes after a newer primary commits via $replacementLabel",
    async ({ replacementLabel, replacementDocument, loadedPattern }) => {
      render(<GroundPlaybackPanel />);
      const primaryInput = screen.getByLabelText(
        "Import strict ground result JSON",
      );
      fireEvent.change(primaryInput, {
        target: {
          files: [
            new File([JSON.stringify(fixture.result)], "primary.json", {
              type: "application/json",
            }),
          ],
        },
      });
      expect(
        await screen.findByText(/Loaded primary.json/),
      ).toBeInTheDocument();

      const pending: FileReader[] = [];
      const readSpy = vi
        .spyOn(FileReader.prototype, "readAsText")
        .mockImplementation(function (this: FileReader) {
          pending.push(this);
        });
      const complete = (reader: FileReader, text: string) => {
        Object.defineProperty(reader, "result", {
          configurable: true,
          value: text,
        });
        reader.onload?.(new ProgressEvent("load") as ProgressEvent<FileReader>);
      };
      const comparison = structuredClone(fixture.result);
      comparison.request_id = "stale-comparison";

      fireEvent.change(screen.getByLabelText(replacementLabel), {
        target: {
          files: [
            new File([replacementDocument], "replacement.json", {
              type: "application/json",
            }),
          ],
        },
      });
      fireEvent.change(
        screen.getByLabelText("Import ground comparison result JSON"),
        {
          target: {
            files: [
              new File([JSON.stringify(comparison)], "comparison.json", {
                type: "application/json",
              }),
            ],
          },
        },
      );
      expect(pending).toHaveLength(2);

      await act(async () => {
        complete(pending[0], replacementDocument);
        await Promise.resolve();
      });
      expect(await screen.findByText(loadedPattern)).toBeInTheDocument();
      await act(async () => {
        complete(pending[1], JSON.stringify(comparison));
        await Promise.resolve();
        await Promise.resolve();
      });
      readSpy.mockRestore();

      expect(
        screen.queryByRole("table", { name: "Ground result comparison table" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("table", {
          name: "Ground comparison trajectory evidence",
        }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("table", {
          name: "Ground comparison event evidence",
        }),
      ).not.toBeInTheDocument();
    },
  );

  it("imports a workspace atomically and exposes deterministic exports", async () => {
    vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(
      () => undefined,
    );
    render(<GroundPlaybackPanel />);
    const comparison = structuredClone(fixture.result);
    comparison.request_id = "workspace-comparison";
    comparison.provenance.input_sha256 = "b".repeat(64);
    comparison.trajectory.forEach((point) => {
      point.time_s += 0.2;
    });
    comparison.events.forEach((event) => {
      event.time_s += 0.2;
    });
    comparison.termination.time_s += 0.2;
    const workspace = {
      schema_version: "rate-of-closure-ground-playback-workspace/v2",
      result: fixture.result,
      comparison: { result: comparison, visible: false },
      playback: { time_s: 1.60466094435, speed: 2, loop: true },
      view: { yaw_deg: -37.5, pitch_deg: 18, zoom: 1.75 },
    };
    fireEvent.change(
      screen.getByLabelText("Import ground playback workspace"),
      {
        target: {
          files: [
            new File([JSON.stringify(workspace)], "session.json", {
              type: "application/json",
            }),
          ],
        },
      },
    );

    expect(
      await screen.findByText(/Loaded workspace session.json/),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Ground playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop ground playback")).toBeChecked();
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent("1.6047 s");
    expect(screen.getByText("workspace-comparison")).toBeInTheDocument();
    expect(screen.getByLabelText("Show comparison overlay")).not.toBeChecked();
    expect(
      screen.getByRole("button", { name: "Export ground result JSON" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Export ground trajectory CSV" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Export ground events CSV" }),
    ).toBeEnabled();

    const retained = screen.getByRole("rowheader", { name: "Carry" });
    fireEvent.click(screen.getByRole("button", { name: "Play ground result" }));
    expect(
      screen.getByRole("button", { name: "Pause ground result" }),
    ).toBeEnabled();
    fireEvent.wheel(screen.getByLabelText("Interactive 3D ground playback"), {
      deltaY: -100,
    });
    const invalid = {
      ...workspace,
      playback: { ...workspace.playback, speed: 3 },
    };
    fireEvent.change(
      screen.getByLabelText("Import ground playback workspace"),
      {
        target: {
          files: [
            new File([JSON.stringify(invalid)], "bad-session.json", {
              type: "application/json",
            }),
          ],
        },
      },
    );
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /last valid playback remains loaded/i,
      ),
    );
    expect(retained).toBeInTheDocument();
    expect(screen.getByLabelText("Ground playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop ground playback")).toBeChecked();
    expect(screen.getByLabelText("Show comparison overlay")).not.toBeChecked();
    expect(screen.getByText("workspace-comparison")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Pause ground result" }),
    ).toBeEnabled();
    expect(screen.getByLabelText("Ground playback absolute time")).toHaveValue(
      "1.60466094435",
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Save ground playback workspace" }),
    );
    const calls = vi.mocked(downloadText).mock.calls;
    const saved = JSON.parse(calls[calls.length - 1]?.[1] ?? "");
    expect(saved.schema_version).toBe(
      "rate-of-closure-ground-playback-workspace/v2",
    );
    expect(saved.comparison.visible).toBe(false);
    expect(saved.comparison.result.request_id).toBe("workspace-comparison");
    expect(saved.playback.time_s).toBe(1.60466094435);
    expect(saved.view).toEqual({
      pitch_deg: 18,
      yaw_deg: -37.5,
      zoom: 1.925,
    });

    fireEvent.change(
      screen.getByLabelText("Import ground playback workspace"),
      {
        target: {
          files: [
            new File([JSON.stringify(workspace)], "valid-reload.json", {
              type: "application/json",
            }),
          ],
        },
      },
    );
    expect(
      await screen.findByText(/Loaded workspace valid-reload.json/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Play ground result" }),
    ).toBeEnabled();
  });
});
