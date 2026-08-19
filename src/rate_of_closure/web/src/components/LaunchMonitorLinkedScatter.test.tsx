import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { LaunchMonitorLinkedScatter } from "./LaunchMonitorLinkedScatter";

it("renders an honest unavailable state through empty and equal axis transitions", () => {
  const rows = [{ x: 1, y: 2 }, { x: 2, y: 3 }, { x: 3, y: 4 }];
  const rendered = render(<LaunchMonitorLinkedScatter rows={rows} xField="x" yField="y"
    selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);
  expect(screen.getByRole("img", { name: /linked scatter plot/ })).toBeVisible();

  rendered.rerender(<LaunchMonitorLinkedScatter rows={rows} xField="" yField="y"
    selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);
  expect(screen.getByText("Select two populated variables.")).toBeVisible();

  rendered.rerender(<LaunchMonitorLinkedScatter rows={rows} xField="y" yField="y"
    selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);
  expect(screen.getByText("Select two populated variables.")).toBeVisible();
});

it("does not inspect retained rows while axes are unavailable", () => {
  const rows = new Array(250_000);
  Object.defineProperty(rows, 0, { get: () => { throw new Error("row inspected"); } });

  render(<LaunchMonitorLinkedScatter rows={rows} xField="" yField="y"
    selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);

  expect(screen.getByText("Select two populated variables.")).toBeVisible();
});

it("renders extreme finite axes and disjoint missingness without invalid SVG", () => {
  const rendered = render(<LaunchMonitorLinkedScatter rows={[
    { x: -Number.MAX_VALUE, y: Number.MAX_VALUE },
    { x: 0, y: 0 },
    { x: Number.MAX_VALUE, y: -Number.MAX_VALUE },
  ]} xField="x" yField="y" selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);
  for (const point of rendered.container.querySelectorAll("circle")) {
    expect(point.getAttribute("cx")).not.toMatch(/NaN|Infinity/);
    expect(point.getAttribute("cy")).not.toMatch(/NaN|Infinity/);
  }

  rendered.rerender(<LaunchMonitorLinkedScatter rows={[
    { x: 1, y: null }, { x: 2, y: null }, { x: 3, y: null },
    { x: null, y: 4 }, { x: null, y: 5 }, { x: null, y: 6 },
  ]} xField="x" yField="y" selectedRawIndex={null} onSelectedRawIndex={vi.fn()} />);
  expect(screen.getByText(/0 available/)).toBeVisible();
});
