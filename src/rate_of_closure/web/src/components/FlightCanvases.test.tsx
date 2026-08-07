import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FlightCanvases } from "./FlightCanvases";
import { spatialTargetFromRegion, DEFAULT_TARGET } from "../model/targets";

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
});
