import { fireEvent, render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { FlightExplorerPanel } from "./FlightExplorerPanel";

beforeAll(() => {
  const ctx: unknown = new Proxy(function () {} as object, {
    get: (_target, prop) =>
      prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
    set: () => true,
    apply: () => ctx,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    ctx as CanvasRenderingContext2D,
  );
});

describe("FlightExplorerPanel input editing", () => {
  it("accepts and preserves a negative spin-axis tilt", () => {
    render(<FlightExplorerPanel />);
    const input = screen.getByLabelText("Spin-Axis Tilt") as HTMLInputElement;

    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "-" } });
    expect(input.value).toBe("-");
    fireEvent.change(input, { target: { value: "-12.5" } });
    fireEvent.blur(input);

    expect(input.value).toBe("-12.5");
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
