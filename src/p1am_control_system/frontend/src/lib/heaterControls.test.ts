import { describe, expect, it } from "vitest";

import {
  setpointOutcome,
  startStopView,
  type HeaterStatusView,
} from "./heaterControls";

const status = (over: Partial<HeaterStatusView>): HeaterStatusView => ({
  state: "idle",
  permissive: false,
  ...over,
});

describe("startStopView", () => {
  it("offers Start (enabled) when stopped/idle", () => {
    expect(startStopView(status({ state: "idle", permissive: false }))).toEqual({
      label: "Start",
      command: "start",
      disabled: false,
    });
  });

  it("offers Stop when armed (started, not yet heating)", () => {
    expect(startStopView(status({ state: "armed", permissive: true }))).toEqual({
      label: "Stop",
      command: "stop",
      disabled: false,
    });
  });

  it("offers Stop when running", () => {
    const v = startStopView(status({ state: "running", permissive: true }));
    expect(v.command).toBe("stop");
    expect(v.label).toBe("Stop");
    expect(v.disabled).toBe(false);
  });

  it("disables the button when tripped (must acknowledge first)", () => {
    const v = startStopView(status({ state: "tripped", permissive: true }));
    expect(v.command).toBe("start");
    expect(v.disabled).toBe(true);
  });

  it("disables the button when status is unknown", () => {
    expect(startStopView(undefined)).toEqual({
      label: "Start",
      command: "start",
      disabled: true,
    });
  });
});

describe("setpointOutcome", () => {
  it("applies immediately when started (permissive)", () => {
    expect(setpointOutcome(status({ state: "running", permissive: true }))).toBe(
      "apply",
    );
    expect(setpointOutcome(status({ state: "armed", permissive: true }))).toBe(
      "apply",
    );
  });

  it("stages when stopped — it will apply on Start", () => {
    expect(setpointOutcome(status({ state: "idle", permissive: false }))).toBe(
      "stage",
    );
  });

  it("blocks when tripped or status unknown", () => {
    expect(setpointOutcome(status({ state: "tripped", permissive: true }))).toBe(
      "blocked",
    );
    expect(setpointOutcome(undefined)).toBe("blocked");
  });
});
