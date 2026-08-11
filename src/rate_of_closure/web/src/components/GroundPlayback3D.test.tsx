import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/ground_reference_pipeline_golden_v1.json";
import { parseFlightToGroundResultRecord } from "../model/flightGroundResultContract";
import { GroundPlaybackTimeline } from "../model/groundPlayback";
import { normalizeGroundPlaybackYawDegrees } from "../model/groundPlaybackWorkspace";
import { GroundPlayback3D } from "./GroundPlayback3D";

describe("GroundPlayback3D", () => {
  const timeline = new GroundPlaybackTimeline(
    parseFlightToGroundResultRecord(fixture.result),
  );

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

  it("exposes phase-aware controls, exact markers, and locked aspect", () => {
    render(<GroundPlayback3D timeline={timeline} />);
    expect(
      screen.getByRole("button", { name: "Play ground result" }),
    ).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Jump to Roll" }));
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent("Roll");
    expect(screen.getByText("Carry / first contact")).toBeInTheDocument();
    expect(screen.getByText("Total / rest")).toBeInTheDocument();
    expect(screen.getByLabelText("Interactive 3D ground playback")).toHaveStyle(
      {
        width: "100%",
        height: "auto",
        aspectRatio: "860 / 420",
      },
    );
  });

  it("steps exact frames and restarts when playing at the end", () => {
    let nextId = 0;
    const callbacks = new Map<number, FrameRequestCallback>();
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      nextId += 1;
      callbacks.set(nextId, callback);
      return nextId;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      callbacks.delete(id);
    });
    render(<GroundPlayback3D timeline={timeline} />);
    fireEvent.click(screen.getByRole("button", { name: "Jump to Rest" }));
    fireEvent.click(screen.getByRole("button", { name: "Play ground result" }));
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent("Impact");
    expect(callbacks.size).toBe(1);
    fireEvent.click(
      screen.getByRole("button", { name: "Pause ground result" }),
    );
    expect(callbacks.size).toBe(0);
    fireEvent.click(
      screen.getByRole("button", { name: "Next exact ground frame" }),
    );
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent("Skid");
  });

  it("loops while retaining one scheduled animation frame", () => {
    let callback: FrameRequestCallback | undefined;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((value) => {
      callback = value;
      return 1;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(
      () => undefined,
    );
    render(<GroundPlayback3D timeline={timeline} />);
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Loop ground playback" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Jump to Rest" }));
    fireEvent.click(screen.getByRole("button", { name: "Play ground result" }));
    if (!callback) throw new Error("Expected one animation callback");
    act(() => callback?.(performance.now() + 20));
    expect(
      screen.getByRole("button", { name: "Pause ground result" }),
    ).toBeEnabled();
  });

  it("restores portable paused playback and camera state", () => {
    const onStateChange = vi.fn();
    render(
      <GroundPlayback3D
        timeline={timeline}
        initialState={{
          playback: { timeS: 1.205, speed: 2, loop: true },
          view: { yawDeg: -37.5, pitchDeg: 18, zoom: 1.75 },
        }}
        onStateChange={onStateChange}
      />,
    );

    expect(screen.getByLabelText("Ground playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop ground playback")).toBeChecked();
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent("1.2050 s");
    expect(
      screen.getByRole("button", { name: "Play ground result" }),
    ).toBeEnabled();
    expect(onStateChange).toHaveBeenCalledWith({
      playback: { timeS: 1.205, speed: 2, loop: true },
      view: { yawDeg: -37.5, pitchDeg: 18, zoom: 1.75 },
    });

    expect(normalizeGroundPlaybackYawDegrees(397.5)).toBe(37.5);
    expect(normalizeGroundPlaybackYawDegrees(-397.5)).toBe(-37.5);
  });

  it("uses the union absolute window and labels a held primary observation", () => {
    const comparisonRecord = structuredClone(fixture.result) as Record<
      string,
      unknown
    >;
    comparisonRecord.request_id = "comparison-run";
    for (const point of comparisonRecord.trajectory as Array<
      Record<string, unknown>
    >) {
      point.time_s = Number(point.time_s) + 0.2;
    }
    for (const event of comparisonRecord.events as Array<
      Record<string, unknown>
    >) {
      event.time_s = Number(event.time_s) + 0.2;
    }
    const termination = comparisonRecord.termination as Record<string, unknown>;
    termination.time_s = Number(termination.time_s) + 0.2;
    const comparisonTimeline = new GroundPlaybackTimeline(
      parseFlightToGroundResultRecord(comparisonRecord),
    );
    const onStateChange = vi.fn();
    render(
      <GroundPlayback3D
        timeline={timeline}
        comparisonTimeline={comparisonTimeline}
        showComparison
        onStateChange={onStateChange}
      />,
    );

    const scrubber = screen.getByLabelText("Ground playback absolute time");
    expect(scrubber).toHaveAttribute(
      "max",
      String(comparisonTimeline.endTimeS),
    );
    fireEvent.change(scrubber, { target: { value: timeline.endTimeS + 0.1 } });
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent(/primary held at rest · comparison active/i);
    expect(onStateChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        playback: expect.objectContaining({ timeS: timeline.endTimeS + 0.1 }),
      }),
    );
    expect(
      screen.getByText(/Dashed path \/ diamonds: comparison/),
    ).toBeInTheDocument();
  });

  it("toggles only comparison artists without changing the union playback session", () => {
    const comparisonRecord = structuredClone(fixture.result) as Record<
      string,
      unknown
    >;
    comparisonRecord.request_id = "comparison-run";
    for (const point of comparisonRecord.trajectory as Array<
      Record<string, unknown>
    >) {
      point.time_s = Number(point.time_s) + 0.2;
    }
    for (const event of comparisonRecord.events as Array<
      Record<string, unknown>
    >) {
      event.time_s = Number(event.time_s) + 0.2;
    }
    const termination = comparisonRecord.termination as Record<string, unknown>;
    termination.time_s = Number(termination.time_s) + 0.2;
    const comparisonTimeline = new GroundPlaybackTimeline(
      parseFlightToGroundResultRecord(comparisonRecord),
    );
    const { rerender } = render(
      <GroundPlayback3D
        timeline={timeline}
        comparisonTimeline={comparisonTimeline}
        showComparison
      />,
    );
    const target = timeline.endTimeS + 0.1;
    fireEvent.change(screen.getByLabelText("Ground playback absolute time"), {
      target: { value: target },
    });

    rerender(
      <GroundPlayback3D
        timeline={timeline}
        comparisonTimeline={comparisonTimeline}
        showComparison={false}
      />,
    );

    const hiddenScrubber = screen.getByLabelText(
      "Ground playback absolute time",
    );
    expect(hiddenScrubber).toHaveAttribute(
      "max",
      String(comparisonTimeline.endTimeS),
    );
    expect(hiddenScrubber).toHaveValue(String(target));
    expect(
      screen.queryByText(/Dashed path \/ diamonds: comparison/),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Ground playback position" }),
    ).toHaveTextContent(/primary held at rest · comparison active/i);

    rerender(
      <GroundPlayback3D
        timeline={timeline}
        comparisonTimeline={comparisonTimeline}
        showComparison
      />,
    );
    expect(screen.getByLabelText("Ground playback absolute time")).toHaveValue(
      String(target),
    );
    expect(
      screen.getByText(/Dashed path \/ diamonds: comparison/),
    ).toBeInTheDocument();
  });
});
