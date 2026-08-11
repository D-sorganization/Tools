import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SCENARIO } from "../model/impact";
import { ClubCanvas } from "./ClubCanvas";

describe("ClubCanvas camera", () => {
  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: (_target, property) => property === "createRadialGradient"
        ? () => ({ addColorStop: () => undefined })
        : property === "measureText" ? () => ({ width: 0 }) : () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  it("snaps all canonical views and suspends moving-head tracking on orbit", () => {
    const view = render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    fireEvent.click(screen.getByRole("button", { name: "Face On" }));
    expect(screen.getByRole("button", { name: "Face On" }))
      .toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Down the Line" }));
    fireEvent.click(screen.getByRole("button", { name: "Overhead" }));
    expect(screen.getByRole("button", { name: "Overhead" }))
      .toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("checkbox", { name: "Track Clubhead" }))
      .toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Auto Fit camera" }))
      .toBeChecked();
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking Clubhead");
    fireEvent.pointerDown(canvas, { pointerId: 3, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 3, clientX: 130, clientY: 120 });
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking suspended");
    fireEvent.click(screen.getByRole("button", { name: "Re-center Clubhead" }));
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking Clubhead");
    view.unmount();
  });
});
