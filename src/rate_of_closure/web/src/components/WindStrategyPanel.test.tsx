import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { WindStrategyPanel } from "./WindStrategyPanel";
import { directLaunch } from "../model/flightExplorer";
import { analyzeWindStrategies, type WindStrategyRequest } from "../model/windUncertainty";
import type { WindStrategyRunController, WindStrategyRunner } from "../model/windStrategyWorkerClient";
import { spatialTargetFromRegion, DEFAULT_TARGET } from "../model/targets";
import {
  createSpatialTarget, sphereTolerance, surfaceCorridorTolerance, targetPointFromFrame,
} from "../model/spatialTarget";

const launch = directLaunch({
  ballSpeedMph: 150,
  launchAngleDeg: 12,
  launchDirectionDeg: 1,
  spinRpm: 2500,
  spinAxisTiltDeg: -4,
});
const target = spatialTargetFromRegion(DEFAULT_TARGET);

describe("WindStrategyPanel", () => {
  it("runs the current launch against the canonical target and exposes all cohorts", async () => {
    let captured: WindStrategyRequest | null = null;
    const runner: WindStrategyRunner = (request, onProgress) => {
      captured = request;
      onProgress({ completed: 1, total: request.uncertainty.trials });
      return {
        promise: Promise.resolve(analyzeWindStrategies(request)),
        cancel: vi.fn(),
      };
    };
    render(<WindStrategyPanel launch={launch} target={target} runner={runner} />);

    expect(screen.getByLabelText("Wind strategy trials")).toBeInTheDocument();
    expect(screen.getByLabelText("Wind strategy seed")).toBeInTheDocument();
    expect(screen.getByLabelText("True wind speed")).toBeInTheDocument();
    expect(screen.getByLabelText("Wind speed estimate bias")).toBeInTheDocument();
    expect(screen.getByLabelText("Wind speed estimate standard deviation")).toBeInTheDocument();
    expect(screen.getByLabelText("Wind estimate correlation")).toBeInTheDocument();
    expect(screen.getByLabelText("Crosswind aim gain")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Wind strategy trials"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("True wind speed"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));

    await waitFor(() => expect(screen.getByRole("status", {
      name: "Wind strategy run status",
    })).toHaveTextContent("Completed 2 trials"));
    expect(captured).not.toBeNull();
    expect(captured!.strategies[0].launch).toEqual(launch);
    expect(captured!.target).toEqual({
      forward_m: target.point.appCoordinatesM[0],
      right_m: target.point.appCoordinatesM[2],
    });
    expect(screen.getByRole("table", { name: "Wind strategy summary" }))
      .toHaveTextContent("Completed");
    const basis = screen.getByRole("region", { name: "Wind strategy calculation basis" });
    expect(basis).toHaveTextContent("waterloo_penner");
    expect(basis).toHaveTextContent("2 / 4199");
    expect(basis).toHaveTextContent(/230\.000 m forward, 0\.000 m right; radius 10\.000 m/);
    expect(basis).toHaveTextContent("0.010000 s step; 10.000 s maximum");
    expect(basis).toHaveTextContent("100.0000");
    expect(basis).toHaveTextContent("0.90000");
    expect(basis).toHaveTextContent(/Current launch: .*rad\/\(m\/s\).*0\.2000 deg\/\(m\/s\)/);
    expect(screen.getByText(/Completed: 2\/2 plotted/)).toBeInTheDocument();
    expect(screen.getByText(/Nonconverged: 0\/0 plotted/)).toBeInTheDocument();
    expect(screen.getByText(/Invalid: 0\/0 plotted/)).toBeInTheDocument();
    const scatterGraphic = screen.getByRole("img", { name: /Wind strategy scatter/ });
    expect(scatterGraphic).toBeInTheDocument();
    expect(within(scatterGraphic).getAllByTestId("wind-scatter-gridline")).toHaveLength(6);
    expect(within(scatterGraphic).getAllByTestId("wind-scatter-tick")).toHaveLength(6);
    expect(within(scatterGraphic).getByTestId("wind-scatter-marks"))
      .toHaveAttribute("clip-path", expect.stringMatching(/^url\(#wind-scatter-clip-.+\)$/));
    expect(screen.getByRole("status", { name: "Wind scatter zoom" })).toHaveTextContent("100%");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in wind scatter" }));
    expect(screen.getByRole("status", { name: "Wind scatter zoom" })).toHaveTextContent("125%");
    fireEvent.change(screen.getByLabelText("Wind scatter horizontal axis"), {
      target: { value: "true_wind_speed_mps" },
    });
    await waitFor(() => expect(screen.getByRole("status", { name: "Wind scatter zoom" }))
      .toHaveTextContent("100%"));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in wind scatter" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom out wind scatter" }));
    expect(screen.getByRole("status", { name: "Wind scatter zoom" })).toHaveTextContent("100%");
    fireEvent.click(screen.getByRole("button", { name: "Auto Fit" }));
    expect(screen.getByRole("status", { name: "Wind scatter zoom" })).toHaveTextContent("100%");
    expect(screen.getByTestId("wind-scatter-legend")).toBeInTheDocument();
    expect(screen.getByText("invalid").closest("g")?.querySelector("circle"))
      .toHaveAttribute("fill", "#ef4444");
    fireEvent.change(screen.getByLabelText("Wind scatter legend position"), {
      target: { value: "bottom-right" },
    });
    expect(screen.getByLabelText("Wind scatter legend position")).toHaveValue("bottom-right");
    fireEvent.click(screen.getByRole("checkbox", { name: "Show wind scatter legend" }));
    expect(screen.queryByTestId("wind-scatter-legend")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export raw wind strategy CSV" }))
      .toBeEnabled();
  });

  it("cancels and terminates an active worker run", () => {
    const cancel = vi.fn();
    const runner: WindStrategyRunner = (): WindStrategyRunController => ({
      promise: new Promise(() => undefined),
      cancel,
    });
    render(<WindStrategyPanel launch={launch} target={target} runner={runner} />);

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel wind strategy analysis" }));

    expect(cancel).toHaveBeenCalledOnce();
    expect(screen.getByRole("status", { name: "Wind strategy run status" }))
      .toHaveTextContent("Cancelled");
  });

  it("shows runner errors without retaining stale output", async () => {
    const runner: WindStrategyRunner = () => ({
      promise: Promise.reject(new Error("worker simulation failed")),
      cancel: vi.fn(),
    });
    render(<WindStrategyPanel launch={launch} target={target} runner={runner} />);

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("worker simulation failed");
    expect(screen.queryByRole("table", { name: "Wind strategy summary" }))
      .not.toBeInTheDocument();
  });

  it("fails closed instead of projecting an aerial waypoint onto the ground", () => {
    const runner = vi.fn<WindStrategyRunner>();
    const aerial = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([120, 22, 0], "app"),
      tolerance: sphereTolerance(3),
      elevationSource: "absolute",
      groundSource: null,
    });
    render(<WindStrategyPanel launch={launch} target={aerial} runner={runner} />);

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));

    expect(screen.getByRole("alert")).toHaveTextContent(/requires.*landing area/i);
    expect(runner).not.toHaveBeenCalled();
  });

  it("uses the conservative inscribed radius for a narrow landing corridor", () => {
    let captured: WindStrategyRequest | null = null;
    const runner: WindStrategyRunner = (request) => {
      captured = request;
      return { promise: new Promise(() => undefined), cancel: vi.fn() };
    };
    const corridor = createSpatialTarget({
      label: "Narrow fairway",
      kind: "landing_area",
      point: targetPointFromFrame([180, 0, 0], "app"),
      tolerance: surfaceCorridorTolerance(12, 2),
      elevationSource: "course_surface",
      groundSource: "flat-ground/v1",
    });
    render(<WindStrategyPanel launch={launch} target={corridor} runner={runner} />);

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));

    expect(captured!.analysis.target_radius_m).toBe(2);
    expect(captured!.analysis.miss_scale_m).toBe(2);
  });

  it("invalidates captured results when launch, target, or wind settings change", async () => {
    const runner: WindStrategyRunner = (request) => ({
      promise: Promise.resolve(analyzeWindStrategies(request)), cancel: vi.fn(),
    });
    const view = render(<WindStrategyPanel launch={launch} target={target} runner={runner} />);
    fireEvent.change(screen.getByLabelText("Wind strategy trials"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));
    expect(await screen.findByRole("table", { name: "Wind strategy summary" })).toBeVisible();

    const movedTarget = createSpatialTarget({
      ...target,
      point: targetPointFromFrame([target.point.appCoordinatesM[0] + 5, 0, 0], "app"),
    });
    view.rerender(<WindStrategyPanel launch={launch} target={movedTarget} runner={runner} />);
    await waitFor(() => expect(screen.queryByRole("table", {
      name: "Wind strategy summary",
    })).not.toBeInTheDocument());
    expect(screen.getByRole("status", { name: "Wind strategy run status" }))
      .toHaveTextContent(/launch or target changed/i);

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));
    expect(await screen.findByRole("table", { name: "Wind strategy summary" })).toBeVisible();
    view.rerender(<WindStrategyPanel launch={{ ...launch, ballSpeedMps: launch.ballSpeedMps + 1 }}
      target={movedTarget} runner={runner} />);
    await waitFor(() => expect(screen.queryByRole("table", {
      name: "Wind strategy summary",
    })).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Run wind strategy analysis" }));
    expect(await screen.findByRole("table", { name: "Wind strategy summary" })).toBeVisible();
    fireEvent.change(screen.getByLabelText("Wind speed estimate bias"), {
      target: { value: "1.25" },
    });
    expect(screen.queryByRole("table", { name: "Wind strategy summary" }))
      .not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Wind strategy run status" }))
      .toHaveTextContent(/wind settings changed/i);
  });
});
