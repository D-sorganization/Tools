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
