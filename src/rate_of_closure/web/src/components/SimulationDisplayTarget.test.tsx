import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { DEFAULT_SCENARIO } from "../model/impact";
import { runSimulation, type SimulationInput } from "../model/simulation";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";
import { defaultViewWorkspace } from "../model/viewWorkspace";

const { flightCanvasSpy } = vi.hoisted(() => ({ flightCanvasSpy: vi.fn() }));
vi.mock("./FlightCanvases", () => ({
  FlightCanvases: (props: unknown) => {
    flightCanvasSpy(props);
    return <div data-testid="flight-canvases" />;
  },
}));

import { SimulationDisplay } from "./SimulationDisplay";

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

describe("SimulationDisplay canonical target rendering", () => {
  it("does not pass the canonical landing target through the legacy overlay too", () => {
    const target = spatialTargetFromRegion(DEFAULT_TARGET);
    const input = {
      sourceKind: "manual", clubheadSpeedMph: 100, omegaDps: [0, 0, 0],
      loftDeg: 10.5, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
      planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
      impactTimeS: null, swingDurationS: 1.5,
      ballSetup: { supportMode: "ground", teeHeightM: 0 },
    } as SimulationInput;
    render(<SimulationDisplay run={null} input={input} scenario={DEFAULT_SCENARIO}
      effectiveLoftDeg={10.5} clubSpec={null} spatialTarget={target}
      onSpatialTargetChange={() => undefined} distanceUnit="yd" />);
    fireEvent.click(screen.getByRole("tab", { name: "Flight" }));

    expect(screen.getByTestId("flight-canvases")).toBeInTheDocument();
    const props = flightCanvasSpy.mock.calls[flightCanvasSpy.mock.calls.length - 1]?.[0];
    expect(props).toEqual(expect.objectContaining({ spatialTarget: target }));
    expect(props).not.toHaveProperty("target");
  });

  it("publishes settled transport state to the serialized workspace owner", async () => {
    const target = spatialTargetFromRegion(DEFAULT_TARGET);
    const input = {
      sourceKind: "manual", clubheadSpeedMph: 100, omegaDps: [0, 0, 0],
      loftDeg: 10.5, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
      planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
      impactTimeS: null, swingDurationS: 1.5,
      ballSetup: { supportMode: "ground", teeHeightM: 0 },
    } as SimulationInput;
    const onWorkspaceChange = vi.fn();
    render(<SimulationDisplay run={runSimulation(input)} input={input}
      scenario={DEFAULT_SCENARIO} effectiveLoftDeg={10.5} clubSpec={null}
      spatialTarget={target} onSpatialTargetChange={() => undefined}
      distanceUnit="yd" viewWorkspace={defaultViewWorkspace}
      onViewWorkspaceChange={onWorkspaceChange} />);

    fireEvent.click(screen.getByRole("checkbox", { name: "Loop" }));
    fireEvent.change(screen.getByRole("slider", { name: "Playback speed" }), {
      target: { value: "0.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "+1 frame" }));

    await waitFor(() => expect(onWorkspaceChange).toHaveBeenCalled());
    const latest = onWorkspaceChange.mock.calls[onWorkspaceChange.mock.calls.length - 1]?.[0];
    expect(latest.playback).toEqual({
      timeS: 0.001,
      playing: false,
      loop: true,
      rate: 0.5,
    });
  });
});
