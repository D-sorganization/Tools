import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FlightCanvases } from "./FlightCanvases";
import { spatialTargetFromRegion, DEFAULT_TARGET } from "../model/targets";
import { planFlightSamples } from "../model/flightSampleInspector";

describe("FlightCanvases responsive layout", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  afterEach(() => vi.restoreAllMocks());

  it.each([
    ["Flight side profile (height vs carry)", "860", "260", "860 / 260"],
    ["Flight top-down view (lateral vs carry)", "860", "220", "860 / 220"],
  ])(
    "preserves the intrinsic aspect ratio of %s at responsive widths",
    (label, width, height, aspectRatio) => {
      render(<FlightCanvases points={[]} />);

      const canvas = screen.getByLabelText(label);
      expect(canvas).toHaveAttribute("width", width);
      expect(canvas).toHaveAttribute("height", height);
      expect(canvas).toHaveStyle({
        width: "100%",
        height: "auto",
        aspectRatio,
      });
    },
  );

  it("redraws both projections when the canonical spatial target changes", () => {
    vi.restoreAllMocks();
    const ellipse = vi.fn();
    const fillText = vi.fn();
    const measureText = vi.fn(() => ({ width: 120 } as TextMetrics));
    const context = new Proxy({ ellipse, fillText, measureText } as unknown as CanvasRenderingContext2D, {
      get: (target, property) => {
        if (property === "ellipse") return ellipse;
        if (property === "fillText") return fillText;
        const existing = Reflect.get(target, property) as unknown;
        return existing ?? vi.fn();
      },
      set: () => true,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
    const points = [
      { time: 0, position: [0, 0, 0] as [number, number, number], velocity: [1, 1, 0] as [number, number, number] },
      { time: 1, position: [100, 0, 0] as [number, number, number], velocity: [1, -1, 0] as [number, number, number] },
    ];
    const firstTarget = spatialTargetFromRegion({ ...DEFAULT_TARGET, distanceM: 140 });
    const view = render(
      <FlightCanvases points={points} showCourse={false} spatialTarget={firstTarget} />,
    );

    expect(ellipse).toHaveBeenCalledTimes(2);
    expect(fillText).toHaveBeenCalledWith(
      expect.stringContaining("ACTIVE · Green Target"),
      expect.any(Number), expect.any(Number),
    );
    const firstProjectedX = ellipse.mock.calls[0][0];
    const movedTarget = spatialTargetFromRegion({ ...DEFAULT_TARGET, distanceM: 180 });
    view.rerender(
      <FlightCanvases points={points} showCourse={false} spatialTarget={movedTarget} />,
    );

    expect(ellipse).toHaveBeenCalledTimes(4);
    expect(ellipse.mock.calls[2][0]).not.toBe(firstProjectedX);
  });

  it.each([[[]], [[{ time: 0, position: [0, 0, 0], velocity: [1, 1, 0] }]]])(
    "renders the active target with a %s-point trajectory",
    (points) => {
      vi.restoreAllMocks();
      const ellipse = vi.fn();
      const fillText = vi.fn();
      const measureText = vi.fn(() => ({ width: 120 } as TextMetrics));
      const context = new Proxy({ ellipse, fillText, measureText } as unknown as CanvasRenderingContext2D, {
        get: (target, property) => Reflect.get(target, property) ?? vi.fn(),
        set: () => true,
      });
      vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
      render(<FlightCanvases points={points as Parameters<typeof FlightCanvases>[0]["points"]}
        showCourse={false} spatialTarget={spatialTargetFromRegion(DEFAULT_TARGET)} />);

      expect(ellipse).toHaveBeenCalledTimes(2);
      expect(fillText).toHaveBeenCalledWith(
        expect.stringContaining("ACTIVE · Green Target"),
        expect.any(Number), expect.any(Number),
      );
      expect(screen.getByLabelText("Flight side profile (height vs carry)"))
        .toHaveAttribute("aria-description", expect.stringContaining("Green Target"));
    },
  );

  it("synchronizes pointer and keyboard selection without selecting the calm ghost", () => {
    const points = [
      { time: 0, position: [0, 0, 0] as [number, number, number], velocity: [1, 1, 0] as [number, number, number] },
      { time: 1, position: [100, 0, 0] as [number, number, number], velocity: [1, -1, 0] as [number, number, number] },
    ];
    const calm = points.map((point) => ({
      ...point,
      position: [point.position[0], 3, point.position[2]] as [number, number, number],
    }));
    const plan = planFlightSamples({ timesS: [0, 1], positionsM: points.map((point) => point.position) });
    const onSelectionChange = vi.fn();
    render(<FlightCanvases points={points} comparisonPoints={calm} plan={plan}
      onSelectionChange={onSelectionChange} />);
    const side = screen.getByLabelText("Flight side profile (height vs carry)");
    vi.spyOn(side, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, width: 860, height: 260, right: 860, bottom: 260,
      x: 0, y: 0, toJSON: () => ({}),
    });
    fireEvent.click(side, { clientX: 34, clientY: 226 });
    expect(onSelectionChange).toHaveBeenLastCalledWith({ cohort: "current", rawIndex: 0 });
    const afterHit = onSelectionChange.mock.calls.length;
    fireEvent.click(side, { clientX: 34, clientY: 203 });
    expect(onSelectionChange).toHaveBeenCalledTimes(afterHit);
    fireEvent.click(side, { clientX: 34, clientY: 10 });
    expect(onSelectionChange).toHaveBeenCalledTimes(afterHit);
    side.focus();
    fireEvent.keyDown(side, { key: "End" });
    expect(onSelectionChange).toHaveBeenLastCalledWith({ cohort: "current", rawIndex: 1 });
    expect(side).toHaveFocus();
    fireEvent.keyDown(side, { key: "Escape" });
    expect(onSelectionChange).toHaveBeenLastCalledWith(null);
  });
});
