import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/launch_monitor_linked_scatter_golden_v1.json";
import { navigateLinkedScatter, planLinkedScatter, projectPlotAxis } from "./launchMonitorLinkedScatter";
import { finiteLaunchMonitorScalar } from "./launchMonitorAnalysisTypes";

describe("launch-monitor linked scatter", () => {
  it("rejects sparse retained row sequences", () => {
    expect(() => planLinkedScatter(new Array(3), "x", "y"))
      .toThrow(/each linked scatter row must be a record/);
  });

  it("matches the Python-owned filtering, decimation, selection, and navigation golden", () => {
    const plain = planLinkedScatter(fixture.rows, fixture.x_field, fixture.y_field, null, fixture.cap);
    const selected = planLinkedScatter(
      fixture.rows, fixture.x_field, fixture.y_field, fixture.selected_raw_index, fixture.cap,
    );
    expect([plain.rawCount, plain.finiteCount]).toEqual([10, 7]);
    expect(plain.points.map((point) => point.rawIndex)).toEqual(fixture.expected.unselected_raw_indices);
    expect(selected.points.map((point) => point.rawIndex)).toEqual(fixture.expected.selected_raw_indices);
    Object.entries(fixture.expected.navigation_from_selected).forEach(([command, expected]) => {
      expect(navigateLinkedScatter(selected, 6, command as "next")).toBe(expected);
    });
    expect(selected.points[2].shotId).toBe("six");
  });

  it("is deeply immutable and clears a selected nonfinite row", () => {
    const plan = planLinkedScatter([{ x: 1, y: 2 }, { x: null, y: 3 }], "x", "y", 1);
    expect(plan.selectedRawIndex).toBeNull();
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.points)).toBe(true);
    expect(Object.isFrozen(plan.points[0])).toBe(true);
  });

  it("caps 100k raw rows at 2000 while retaining selection", () => {
    const rows = Array.from({ length: 100_000 }, (_, index) => ({ x: index, y: index % 97 }));
    const plan = planLinkedScatter(rows, "x", "y", 50_001);
    expect(plan.displayedCount).toBe(2_000);
    expect(plan.points.some((point) => point.rawIndex === 50_001)).toBe(true);
  });

  it("shares strict decimal-string parsing without Number hex coercion", () => {
    expect(fixture.numeric_grammar.values.map(finiteLaunchMonitorScalar)
      .flatMap((value, index) => value === null ? [] : [index]))
      .toEqual(fixture.numeric_grammar.finite_indices);
    const rows = fixture.numeric_grammar.values.map((value, index) => ({ x: value, y: index + 1 }));
    const plan = planLinkedScatter(rows, "x", "y");
    expect(plan.points.map(({ rawIndex }) => rawIndex)).toEqual(fixture.numeric_grammar.finite_indices);
  });

  it("projects extreme, constant, signed-zero, and subnormal axes to finite positions", () => {
    for (const { values, expected } of Object.values(fixture.plot_projection_cases)) {
      const coordinates = projectPlotAxis(values).coordinates;
      expect(coordinates).toEqual(expected);
      expect(coordinates.every((value) => value >= -1 && value <= 1)).toBe(true);
    }
  });
});
