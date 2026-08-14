import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { spatialTargetFromRegion, DEFAULT_TARGET } from "../model/targets";
import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "../model/spatialTarget";
import { SpatialTargetSection } from "./SpatialTargetSection";

describe("SpatialTargetSection", () => {
  it("applies an accessible aerial target without silently coercing fields", () => {
    const onChange = vi.fn();
    render(
      <SpatialTargetSection
        target={spatialTargetFromRegion(DEFAULT_TARGET)}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Target type"), {
      target: { value: "aerial_waypoint" },
    });
    fireEvent.change(screen.getByLabelText("Target downrange m"), {
      target: { value: "175" },
    });
    fireEvent.change(screen.getByLabelText("Target elevation m"), {
      target: { value: "32" },
    });
    fireEvent.change(screen.getByLabelText("Target right offset m"), {
      target: { value: "-4.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));

    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange.mock.calls[0][0]).toMatchObject({
      kind: "aerial_waypoint",
      point: { appCoordinatesM: [175, 32, -4.5] },
      tolerance: { kind: "sphere", radiusM: 10 },
      frame: "app",
      units: "m",
    });
  });

  it("reports invalid coordinates and preserves the last valid target", () => {
    const onChange = vi.fn();
    render(
      <SpatialTargetSection
        target={spatialTargetFromRegion(DEFAULT_TARGET)}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Target downrange m"), {
      target: { value: "not-a-number" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Downrange must be a finite number",
    );
    expect(screen.getByLabelText("Target downrange m")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
    expect(onChange).not.toHaveBeenCalled();
  });

  it("loads the versioned canonical JSON contract", () => {
    const onChange = vi.fn();
    render(
      <SpatialTargetSection
        target={spatialTargetFromRegion(DEFAULT_TARGET)}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Spatial target JSON"), {
      target: {
        value: JSON.stringify({
          schema: "swing_sim.spatial_target",
          schema_version: 1,
          units: "m",
          frame: "app",
          source_frame: "flight",
          label: "Window",
          kind: "aerial_waypoint",
          position_m: { x: 150, elevation: 20, right: 3 },
          tolerance: {
            kind: "box",
            half_extents_m: { x: 4, elevation: 2, right: 3 },
          },
          elevation_source: "absolute",
          ground_source: null,
        }),
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load target JSON" }));

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      label: "Window",
      kind: "aerial_waypoint",
      point: { appCoordinatesM: [150, 20, 3], sourceFrame: "flight" },
      tolerance: { kind: "box", halfExtentsM: [4, 2, 3] },
    }));
  });

  it("converts entry-frame coordinates without moving the canonical target", () => {
    const onChange = vi.fn();
    const initial = spatialTargetFromRegion({ ...DEFAULT_TARGET, lateralM: 12 });
    render(<SpatialTargetSection target={initial} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText("Target coordinate frame"), {
      target: { value: "flight" },
    });

    expect(screen.getByLabelText("Target downrange m")).toHaveValue("230");
    expect(screen.getByLabelText("Target left offset m")).toHaveValue("-12");
    expect(screen.getByLabelText("Target elevation m")).toHaveValue("0");
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));
    expect(onChange.mock.calls[0][0].point.appCoordinatesM).toEqual([230, 0, 12]);
  });

  it("associates an error, focuses its field, and supports keyboard submission", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <SpatialTargetSection target={spatialTargetFromRegion(DEFAULT_TARGET)} onChange={onChange} />,
    );
    const downrange = screen.getByLabelText("Target downrange m");
    await user.clear(downrange);
    await user.type(downrange, "invalid{Enter}");

    const error = screen.getByRole("alert");
    expect(downrange).toHaveFocus();
    expect(downrange).toHaveAttribute("aria-errormessage", error.id);
    expect(error).toHaveTextContent("Downrange must be a finite number");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("describes aerial evaluation as a continuous interpolated passage", () => {
    const target = createSpatialTarget({
      label: "Between samples",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([5, 0, 0], "app"),
      tolerance: sphereTolerance(0.5),
      elevationSource: "absolute",
    });
    render(<SpatialTargetSection target={target} onChange={() => undefined}
      flightPoints={[
        { time: 0, position: [0, 0, 0], velocity: [1, 0, 0] },
        { time: 2, position: [10, 0, 0], velocity: [1, 0, 0] },
      ]} />);
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent(/continuous interpolated nearest passage at 1\.000 s/i);
  });

  it("evaluates landing contact on the course surface, not ball-center elevation", () => {
    render(<SpatialTargetSection target={spatialTargetFromRegion({
      ...DEFAULT_TARGET, distanceM: 100, radiusM: 2,
    })} onChange={() => undefined} flightPoints={[
      { time: 0, position: [0, 0.06, 0], velocity: [1, 1, 0] },
      { time: 4, position: [100, 0.06, 0], velocity: [1, -1, 0] },
    ]} />);
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent(/accepted at the projected course-surface landing contact/i);
  });
});
