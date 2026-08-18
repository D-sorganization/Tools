// @vitest-environment jsdom

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { PuttResult } from "../model/putting";
import { planPuttingSamples, puttingSampleSource } from "../model/puttingSampleInspector";
import { PuttingVisuals } from "./PuttingVisuals";

const RESULT: PuttResult = {
  pathXM: [0, 1, 2, 3], pathYM: [0, 0.5, 0.25, 0], speedsMps: [2, 1.5, 1, 0],
  timesS: [0, 0.1, 0.2, 0.3], skidEndIndex: 2, skidDistanceM: 2,
  totalDistanceM: 3.2, timeS: 0.3, breakM: 0, holed: false,
  speedAtHoleMps: null, marginMps: null, missDistanceM: 0.1,
};

function view(selectedRawIndex: number | null, onSelectionChange = vi.fn()) {
  return render(<PuttingVisuals result={RESULT} plan={planPuttingSamples(puttingSampleSource(RESULT))}
    selectedRawIndex={selectedRawIndex} onSelectionChange={onSelectionChange}
    holeX={3} grade={0} aspect={90} />);
}

describe("PuttingVisuals synchronized sample inspector", () => {
  it("exposes one named focusable visual with exact keyboard navigation", () => {
    const onSelectionChange = vi.fn();
    view(null, onSelectionChange);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    expect(path).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(path, { key: "Home" });
    fireEvent.keyDown(path, { key: "End" });
    fireEvent.keyDown(path, { key: "Escape" });
    expect(onSelectionChange.mock.calls.map(([value]) => value)).toEqual([0, 3, null]);
  });

  it("renders the same selected raw sample on path and speed plots with polite status", () => {
    view(2);
    expect(screen.getAllByTestId("putting-selected-sample")).toHaveLength(2);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Source sample 2");
    expect(status).toHaveTextContent("t 0.200 s");
    expect(status).toHaveTextContent("pure roll");
    expect(status).toHaveTextContent("x 2.000 m");
    expect(status).toHaveTextContent("speed 1.000 m/s");
  });

  it("selects within the rendered pixel hit radius with a lower-index tie", () => {
    const onSelectionChange = vi.fn();
    view(null, onSelectionChange);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    let renderedScale = 0.5;
    Object.defineProperty(path, "getScreenCTM", { value: () => ({
      a: renderedScale, b: 0, c: 0, d: renderedScale, e: 0, f: 0,
    }) });
    for (const scale of [0.5, 1]) {
      renderedScale = scale;
      const sampleX = 46.8292682927 * scale;
      const sampleY = 160 * scale;
      fireEvent(path, new MouseEvent("pointerdown", {
        bubbles: true, clientX: sampleX + 10, clientY: sampleY,
      }));
      expect(onSelectionChange).toHaveBeenLastCalledWith(0);
      fireEvent(path, new MouseEvent("pointerdown", {
        bubbles: true, clientX: sampleX + 13, clientY: sampleY,
      }));
      expect(onSelectionChange).toHaveBeenLastCalledWith(null);
    }
  });
});
