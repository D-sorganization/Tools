import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { defaultCameraState } from "../model/cameraCommands";
import { CameraControlBar } from "./CameraControlBar";

describe("CameraControlBar", () => {
  it("exposes stable keyboard controls, explicit face side, and active state", () => {
    const onPreset = vi.fn();
    const onFaceOnSide = vi.fn();
    const onTracking = vi.fn();
    const onAutoFit = vi.fn();
    const onRecenter = vi.fn();
    const state = { ...defaultCameraState(), presetId: "camera.view.down_the_line" as const };
    render(<CameraControlBar state={state} subjectLabel="Clubhead"
      onPreset={onPreset} onFaceOnSide={onFaceOnSide} onTracking={onTracking}
      onAutoFit={onAutoFit} onRecenter={onRecenter} />);

    const face = screen.getByRole("button", { name: "Face On" });
    const downLine = screen.getByRole("button", { name: "Down the Line" });
    const overhead = screen.getByRole("button", { name: "Overhead" });
    expect(downLine).toHaveAttribute("aria-pressed", "true");
    for (const button of [face, downLine, overhead]) {
      expect(button).toHaveAttribute("data-camera-command");
      expect(button).toHaveAttribute("title", expect.stringContaining("App frame"));
    }
    fireEvent.click(overhead);
    expect(onPreset).toHaveBeenCalledWith("camera.view.overhead");
    fireEvent.change(screen.getByRole("combobox", { name: "Face-on camera side" }), {
      target: { value: "left" },
    });
    expect(onFaceOnSide).toHaveBeenCalledWith("left");
    fireEvent.click(screen.getByRole("checkbox", { name: "Track Clubhead" }));
    expect(onTracking).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("checkbox", { name: "Auto Fit camera" }));
    expect(onAutoFit).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "Re-center Clubhead" }));
    expect(onRecenter).toHaveBeenCalledOnce();
  });

  it("announces a suspended manual override and offers one-action recovery", () => {
    render(<CameraControlBar
      state={{ ...defaultCameraState(), trackingEnabled: true, trackingSuspended: true }}
      subjectLabel="Ball" onPreset={() => undefined} onFaceOnSide={() => undefined}
      onTracking={() => undefined} onAutoFit={() => undefined} onRecenter={() => undefined} />);
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking suspended by manual orbit");
    expect(screen.getByRole("button", { name: "Re-center Ball" })).toBeEnabled();
  });
});
