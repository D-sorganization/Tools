import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SCENARIO } from "../model/impact";
import { CAMERA_COMMAND_IDS } from "../model/cameraPresets";
import { ClubCanvas } from "./ClubCanvas";

describe("ClubCanvas canonical camera controls", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLCanvasElement.prototype, "setPointerCapture", {
      configurable: true,
      value: vi.fn(),
    });
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

  it("exposes stable accessible commands and a keyboard-focusable canvas", () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    const controls = screen.getByRole("group", { name: "Clubhead camera controls" });
    for (const id of CAMERA_COMMAND_IDS) {
      expect(controls.querySelector(`[data-camera-command="${id}"]`)).not.toBeNull();
    }
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    expect(canvas).toHaveAttribute("tabindex", "0");
    canvas.focus();
    expect(canvas).toHaveFocus();
  });

  it("snaps exactly, selects the face side, and preserves zoom through reset", () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    fireEvent.wheel(canvas, { deltaY: -1 });
    expect(canvas).toHaveAttribute("data-camera-zoom", "1.100000");
    fireEvent.click(screen.getByRole("button", { name: "Down the Line" }));
    expect(canvas).toHaveAttribute("data-camera-yaw", "0.000000000000");
    expect(canvas).toHaveAttribute("data-camera-pitch", "0.000000000000");
    expect(screen.getByRole("button", { name: "Down the Line" }))
      .toHaveAttribute("aria-pressed", "true");
    fireEvent.change(screen.getByRole("combobox", { name: "Face-on camera side" }), {
      target: { value: "left" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Face On" }));
    expect(canvas).toHaveAttribute("data-camera-yaw", (Math.PI / 2).toFixed(12));
    fireEvent.click(screen.getByRole("button", { name: "Reset View" }));
    expect(canvas).toHaveAttribute("data-camera-zoom", "1.100000");
    expect(screen.getByRole("button", { name: "Isometric" }))
      .toHaveAttribute("aria-pressed", "true");
  });

  it("preserves a manual orbit while changing zoom", () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    fireEvent(canvas, new MouseEvent("pointerdown", {
      bubbles: true, clientX: 100, clientY: 100,
    }));
    fireEvent(canvas, new MouseEvent("pointermove", {
      bubbles: true, clientX: 130, clientY: 120,
    }));
    expect(canvas).toHaveAttribute("data-camera-view", "custom");
    for (const label of ["Isometric", "Face On", "Down the Line", "Overhead"]) {
      expect(screen.getByRole("button", { name: label }))
        .toHaveAttribute("aria-pressed", "false");
    }
    const yaw = canvas.getAttribute("data-camera-yaw");
    const pitch = canvas.getAttribute("data-camera-pitch");
    fireEvent.wheel(canvas, { deltaY: -1 });
    expect(canvas).toHaveAttribute("data-camera-yaw", yaw);
    expect(canvas).toHaveAttribute("data-camera-pitch", pitch);
    expect(canvas).toHaveAttribute("data-camera-view", "custom");
    fireEvent.click(screen.getByRole("button", { name: "Face On" }));
    fireEvent(canvas, new MouseEvent("pointermove", {
      bubbles: true, clientX: 160, clientY: 140,
    }));
    expect(canvas).toHaveAttribute("data-camera-view", "custom");
    fireEvent.change(screen.getByRole("combobox", { name: "Face-on camera side" }), {
      target: { value: "left" },
    });
    expect(screen.getByRole("button", { name: "Face On" }))
      .toHaveAttribute("aria-pressed", "true");
    expect(canvas).toHaveAttribute("data-camera-view", "camera.view.face_on");
    expect(canvas).toHaveAttribute("data-camera-yaw", (Math.PI / 2).toFixed(12));
    fireEvent.click(screen.getByRole("button", { name: "Overhead" }));
    expect(screen.getByRole("button", { name: "Overhead" }))
      .toHaveAttribute("aria-pressed", "true");
  });

  it("tracks with visible suspension and explicit zoom-preserving recenter", async () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} initialPhase={0} />);
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    fireEvent.wheel(canvas, { deltaY: -1 });
    fireEvent.click(screen.getByRole("checkbox", { name: "Track Clubhead" }));
    await waitFor(() => expect(canvas)
      .toHaveAttribute("data-camera-tracking-state", "camera.tracking.active"));
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking Clubhead");
    expect(canvas).toHaveAttribute("data-camera-zoom", "1.100000");
    fireEvent(canvas, new MouseEvent("pointerdown", {
      bubbles: true, clientX: 100, clientY: 100,
    }));
    fireEvent(canvas, new MouseEvent("pointermove", {
      bubbles: true, clientX: 130, clientY: 120,
    }));
    expect(canvas).toHaveAttribute(
      "data-camera-tracking-state", "camera.tracking.suspended",
    );
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking suspended");
    fireEvent.click(screen.getByRole("button", { name: "Re-center Clubhead" }));
    expect(canvas).toHaveAttribute(
      "data-camera-tracking-state", "camera.tracking.active",
    );
    expect(canvas.getAttribute("data-camera-target"))
      .toBe(canvas.getAttribute("data-camera-subject"));
    expect(canvas).toHaveAttribute("data-camera-zoom", "1.100000");
  });

  it.each([0, 0.5, 1])(
    "centers the tracked clubhead at phase %s without changing zoom",
    async (phase) => {
      render(<ClubCanvas scenario={DEFAULT_SCENARIO} initialPhase={phase} />);
      fireEvent.click(screen.getByRole("button", { name: "Pause" }));
      const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
      fireEvent.wheel(canvas, { deltaY: -1 });
      fireEvent.click(screen.getByRole("checkbox", { name: "Track Clubhead" }));
      await waitFor(() => expect(canvas.getAttribute("data-camera-target"))
        .toBe(canvas.getAttribute("data-camera-subject")));
      expect(canvas).toHaveAttribute("data-camera-zoom", "1.100000");
    },
  );

  it("keeps tracking and fallback state isolated per canvas", async () => {
    render(<><ClubCanvas scenario={DEFAULT_SCENARIO} /><ClubCanvas scenario={DEFAULT_SCENARIO} /></>);
    const track = screen.getAllByRole("checkbox", { name: "Track Clubhead" });
    const fallback = screen.getAllByRole("checkbox", { name: "Auto Fit fallback" });
    fireEvent.click(track[0]);
    fireEvent.click(fallback[0]);
    const canvases = screen.getAllByRole("img", { name: /Animated 3D clubhead/i });
    await waitFor(() => expect(canvases[0])
      .toHaveAttribute("data-camera-tracking-state", "camera.tracking.active"));
    expect(canvases[0]).toHaveAttribute("data-camera-auto-fit-fallback", "true");
    expect(canvases[1]).toHaveAttribute(
      "data-camera-tracking-state", "camera.tracking.off",
    );
    expect(canvases[1]).toHaveAttribute("data-camera-auto-fit-fallback", "false");
  });

  it("applies fallback before rasterizing the first frame after a mode change", async () => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} />);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Auto Fit fallback" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Clubhead display mode" }), {
      target: { value: "Head Fixed in Place" },
    });
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    await waitFor(() => expect(canvas)
      .toHaveAttribute("data-camera-rendered-subject-fits", "true"));
    expect(canvas.getAttribute("data-camera-rendered-zoom"))
      .toBe(canvas.getAttribute("data-camera-zoom"));
  });

  it.each([
    ["Head Fixed in Place", 0.0],
    ["Head Fixed in Place", 0.5],
    ["Head Fixed in Place", 1.0],
    ["Head Moving Through Space", 0.0],
    ["Head Moving Through Space", 0.5],
    ["Head Moving Through Space", 1.0],
  ])("Auto Fit bounds a driver for %s at phase %s", (mode, phase) => {
    render(<ClubCanvas scenario={DEFAULT_SCENARIO} initialPhase={phase} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Clubhead display mode" }), {
      target: { value: mode },
    });
    const canvas = screen.getByRole("img", { name: /Animated 3D clubhead/i });
    fireEvent.wheel(canvas, { deltaY: -1 });
    fireEvent.click(screen.getByRole("button", { name: "Auto Fit" }));
    expect(canvas).toHaveAttribute("data-camera-subject-fits", "true");
  });
});
