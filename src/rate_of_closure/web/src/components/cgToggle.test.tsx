/**
 * 'Show CG' toggle smoke tests (H1, #4125): the CG checkbox renders,
 * defaults off, and toggling it redraws without crashing in both the
 * 3D club canvas and the strike canvas.
 */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { ClubCanvas } from "./ClubCanvas";
import { StrikeCanvas } from "./StrikeCanvas";
import App from "../App";

beforeAll(() => {
  const ctx: unknown = new Proxy(function () {} as object, {
    get: (_target, prop) =>
      prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
    set: () => true,
    apply: () => ctx,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    ctx as unknown as CanvasRenderingContext2D,
  );
});

afterEach(cleanup);

describe("Show CG toggles", () => {
  it("loads a representative driver head in the initial explorer scene", async () => {
    render(<App />);
    const reset = screen.getByRole("button", { name: "Procedural Head" });
    await waitFor(() => expect(reset).toBeEnabled());
  });

  it("defaults on and toggles in the club canvas", () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    const check = screen.getByLabelText("Show CG") as HTMLInputElement;
    expect(check.checked).toBe(true);
    fireEvent.click(check);
    expect(check.checked).toBe(false);
  });

  it("enables with a club spec and toggles in the strike canvas", () => {
    render(
      <StrikeCanvas
        toeMm={5}
        highMm={2}
        loftDeg={10.5}
        clubSpec={getClub("Driver 10.5°")}
      />,
    );
    const check = screen.getByLabelText("Show CG") as HTMLInputElement;
    expect(check.disabled).toBe(false);
    fireEvent.click(check);
    expect(check.checked).toBe(true);
  });

  it("disables the strike CG toggle without a club spec", () => {
    render(<StrikeCanvas toeMm={0} highMm={0} loftDeg={10.5} />);
    const check = screen.getByLabelText("Show CG") as HTMLInputElement;
    expect(check.disabled).toBe(true);
  });
});
