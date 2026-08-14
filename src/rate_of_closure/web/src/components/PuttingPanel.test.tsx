// @vitest-environment jsdom

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PuttingPanel } from "./PuttingPanel";
import { simulatePutt } from "../model/putting";

describe("PuttingPanel GreenConditions bounds", () => {
  it.each([
    ["Clubhead speed m/s", "0.2", "6"],
    ["Green speed (stimp) ft", "3", "16"],
    ["Slope grade %", "0", "10"],
    ["Downhill direction °", "-360", "360"],
    ["Distance to hole m", "0.1", "40"],
  ])("exposes Python-parity bounds for %s", (name, min, max) => {
    render(<PuttingPanel />);

    expect(screen.getByRole("textbox", { name })).toHaveAttribute("min", min);
    expect(screen.getByRole("textbox", { name })).toHaveAttribute("max", max);
  });

  it("preserves exact selection for presentation units and clears on scientific replacement", () => {
    const { rerender } = render(<PuttingPanel distanceUnit="m" />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    expect(screen.getByRole("status")).toHaveTextContent("Source sample 0");

    rerender(<PuttingPanel distanceUnit="yd" />);
    expect(screen.getByRole("status")).toHaveTextContent("Source sample 0");

    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));
    expect(screen.getByRole("status")).toHaveTextContent("No trajectory sample selected");
  });

  it("retains the exact accepted inspector when a scientific replacement fails", () => {
    const executeStudy: typeof simulatePutt = (launch, green, holeDistanceM) => {
      if (green.gradePercent === 1) throw new Error("solver authority unavailable");
      return simulatePutt(launch, green, holeDistanceM);
    };
    render(<PuttingPanel executeStudy={executeStudy} />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    const marker = screen.getAllByTestId("putting-selected-sample")[0];
    const acceptedGeometry = path.innerHTML;
    const acceptedContext = screen.getByLabelText("Displayed putting result context").textContent;

    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));

    expect(screen.getByRole("alert")).toHaveTextContent("solver authority unavailable");
    expect(screen.getByRole("status")).toHaveTextContent("Source sample 0");
    expect(screen.getAllByTestId("putting-selected-sample")[0]).toBe(marker);
    expect(path.innerHTML).toBe(acceptedGeometry);
    expect(screen.getByLabelText("Displayed putting result context")).toHaveTextContent(
      acceptedContext ?? "",
    );
  });

  it("labels a first computation failure without fabricating prior evidence", () => {
    const executeStudy: typeof simulatePutt = () => {
      throw new Error("solver authority unavailable");
    };
    render(<PuttingPanel executeStudy={executeStudy} />);

    expect(screen.getByRole("alert")).toHaveTextContent("No accepted putt is available");
    expect(screen.queryByRole("img", { name: /interactive putt path/i })).toBeNull();
    expect(screen.queryByText(/prior accepted putt remains displayed/i)).toBeNull();
  });

  it("snapshots mutable results and clears selection for a same-object replacement", () => {
    const cached = simulatePutt(
      { ballSpeedMps: 2, launchAngleDeg: 1, horizontalSpeedMps: 2,
        spinRadS: -1, effectiveLoftDeg: 1 },
      { stimpFt: 10, gradePercent: 0, aspectDeg: 90 }, 3,
    );
    const executeStudy: typeof simulatePutt = () => cached;
    const { rerender } = render(<PuttingPanel executeStudy={executeStudy} />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    const rollout = screen.getByRole("button", { name: /Roll-Out Distance/ });
    const displayed = rollout.textContent;
    const originalDistance = cached.totalDistanceM;
    cached.totalDistanceM = 999;
    rerender(<PuttingPanel executeStudy={executeStudy} distanceUnit="yd" />);
    expect(screen.getByRole("button", { name: /Roll-Out Distance/ })).toHaveTextContent(
      displayed ?? "",
    );
    cached.totalDistanceM = originalDistance;
    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));
    expect(screen.getByRole("status")).toHaveTextContent("No trajectory sample selected");
  });
});
