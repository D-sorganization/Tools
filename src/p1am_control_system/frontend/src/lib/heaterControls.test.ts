import { describe, expect, it } from "vitest";

import {
  resolveStartTarget,
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

describe("resolveStartTarget", () => {
  it("uses the operator's typed positive value", () => {
    expect(resolveStartTarget("650.0", 700)).toBe(650);
  });

  it("falls back to the recalled setpoint when the box is '0' (startup race)", () => {
    // The entry box defaults to "0" before the first telemetry frame pre-fills
    // it; Start must still apply the recalled/held target instead of only arming.
    expect(resolveStartTarget("0", 700)).toBe(700);
  });

  it("falls back when the box is empty or non-numeric", () => {
    expect(resolveStartTarget("", 500)).toBe(500);
    expect(resolveStartTarget("   ", 500)).toBe(500);
    expect(resolveStartTarget("abc", 500)).toBe(500);
  });

  it("returns null when neither the box nor the fallback is positive", () => {
    expect(resolveStartTarget("0", null)).toBeNull();
    expect(resolveStartTarget("0", undefined)).toBeNull();
    expect(resolveStartTarget("0", 0)).toBeNull();
    expect(resolveStartTarget("", null)).toBeNull();
  });

  it("prefers a positive typed value over the fallback", () => {
    expect(resolveStartTarget("300", 700)).toBe(300);
  });

  it("ignores a non-positive typed value and a non-positive fallback", () => {
    expect(resolveStartTarget("-50", 700)).toBe(700); // negative typed -> fallback
    expect(resolveStartTarget("-50", -10)).toBeNull(); // both non-positive -> null
  });

  it("treats a non-finite fallback as absent", () => {
    expect(resolveStartTarget("0", Number.NaN)).toBeNull();
    expect(resolveStartTarget("0", Number.POSITIVE_INFINITY)).toBeNull();
  });

  it("throws TypeError when stagedText is not a string (DbC)", () => {
    // @ts-expect-error deliberate wrong type to exercise the runtime guard
    expect(() => resolveStartTarget(650, 700)).toThrow(TypeError);
  });
});
