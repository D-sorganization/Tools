import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import {
  PowerSupplyControl,
  approachingLimit,
  parseConfigNumber,
  type PowerSupplyConfig,
  type PowerSupplyStatus,
  type NoiseStats,
} from "./PowerSupplyControl";

/**
 * Power-supply setpoint entry and alarm cues (#4020, #4042b).
 *
 * #4020: `stagedSetpointText` was initialised to "0" and NEVER seeded from
 * `liveStatus.setpoint_a`, while `nudgeSetpoint` computed `staged + delta` and
 * COMMANDED it immediately. So with the supply running at 30 A, an operator
 * reloading the kiosk saw "0" next to a telemetry readout of 30.00 A, and one
 * tap of "+" commanded 1 A — collapsing the output by 29 A with no
 * confirmation. That also contradicted the file's own documented contract:
 * "Apply commits the staged value — typing alone commands nothing."
 *
 * #4042b: the approaching-alarm highlights were computed from `configDraft`,
 * the LOCAL UNCOMMITTED draft, and `ConfigField` wrote `parseFloat` straight
 * into it — so clearing the field stored NaN and `x >= 0.9 * NaN` is false,
 * switching the pre-alarm cue OFF while the supply was still climbing.
 */

const noise: NoiseStats = {
  sample_count: 10,
  mean: 30,
  std: 0.1,
  peak_to_peak: 0.3,
  rms_about_mean: 0.1,
  coeff_of_variation: 0.003,
  metric: "std",
  metric_value: 0.1,
  threshold: null,
  arcing: false,
};

const config: PowerSupplyConfig = {
  command_tag: "TAG_10",
  current_feedback_tag: "TAG_12",
  voltage_feedback_tag: "TAG_13",
  temp_tag: "TAG_14",
  command_label: "Cmd",
  aux_command_label: "Aux",
  current_feedback_label: "Current",
  voltage_feedback_label: "Voltage",
  temp_label: "Temp",
  current_full_scale_a: 100,
  voltage_full_scale_v: 60,
  current_setpoint_min_a: 0,
  current_setpoint_max_a: 100,
  power_alarm_max_w: 1000,
  temp_alarm_max_c: 100,
  setpoint_ramp_rate_pct_per_s: 5,
  output_clamp_percent: 100,
  noise_window: 20,
  noise_metric: "std",
  current_arc_threshold: null,
  voltage_arc_threshold: null,
};

const runningAt30A: PowerSupplyStatus = {
  state: "running",
  mode: "current",
  permissive: true,
  setpoint_a: 30,
  setpoint_w: null,
  measured_current_a: 30.0,
  measured_voltage_v: 20.0,
  measured_power_w: 600,
  measured_temp_c: 40,
  commanded_output_percent: 30,
  trips: [],
  output_clamp_percent: 100,
  output_clamped: false,
  effective_max_current_a: 100,
  current_noise: noise,
  voltage_noise: noise,
  arcing: false,
};

let fetchMock: ReturnType<typeof vi.fn<[RequestInfo | URL], Promise<Response>>>;

beforeEach(() => {
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/power_supply/config")) {
      return { ok: true, json: async () => config } as unknown as Response;
    }
    return {
      ok: true,
      json: async () => ({ applied_a: 0 }),
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Every POST the component made to the setpoint endpoint. */
function setpointWrites(): unknown[] {
  return fetchMock.mock.calls.filter((c) =>
    String(c[0]).includes("/api/power_supply/setpoint"),
  );
}

const setpointInput = () =>
  document.querySelector(".ps-setpoint-input") as HTMLInputElement;

describe("PowerSupplyControl setpoint staging (#4020)", () => {
  it("seeds the staged entry from the live setpoint instead of a stale 0", async () => {
    render(<PowerSupplyControl liveStatus={runningAt30A} />);
    await waitFor(() => expect(setpointInput()).toBeTruthy());
    // The kiosk-reload scenario: telemetry says 30.00 A, so the box must too.
    await waitFor(() => expect(setpointInput().value).toBe("30.00"));
  });

  it("+/- adjust the STAGED value only and command nothing", async () => {
    render(<PowerSupplyControl liveStatus={runningAt30A} />);
    await waitFor(() => expect(setpointInput().value).toBe("30.00"));

    const plus = screen.getByTitle(/Increase by/);
    await act(async () => {
      plus.click();
    });

    // Staged moved from the REAL running value, not from 0 …
    expect(setpointInput().value).toBe("31.00");
    // … and nothing was commanded: Apply is the sole write path, exactly as
    // the component's documented contract says.
    expect(setpointWrites()).toHaveLength(0);
  });

  it("stops re-seeding once the operator has staged a value", async () => {
    const { rerender } = render(<PowerSupplyControl liveStatus={runningAt30A} />);
    await waitFor(() => expect(setpointInput().value).toBe("30.00"));

    const minus = screen.getByTitle(/Decrease by/);
    await act(async () => {
      minus.click();
    });
    expect(setpointInput().value).toBe("29.00");

    // A later frame with a different setpoint must not yank the staged entry.
    rerender(
      <PowerSupplyControl liveStatus={{ ...runningAt30A, setpoint_a: 30.5 }} />,
    );
    expect(setpointInput().value).toBe("29.00");
  });

  it("never stages below zero", async () => {
    render(<PowerSupplyControl liveStatus={{ ...runningAt30A, setpoint_a: 0.5 }} />);
    await waitFor(() => expect(setpointInput().value).toBe("0.50"));
    const minus = screen.getByTitle(/Decrease by/);
    await act(async () => {
      minus.click();
    });
    expect(setpointInput().value).toBe("0.00");
  });

  it("Apply is what writes the staged value", async () => {
    render(<PowerSupplyControl liveStatus={runningAt30A} />);
    await waitFor(() => expect(setpointInput().value).toBe("30.00"));

    await act(async () => {
      screen.getByText("Apply Setpoint").click();
    });
    expect(setpointWrites()).toHaveLength(1);
  });
});

describe("approachingLimit (#4042b)", () => {
  it("flags a measurement at or past 90 % of the limit", () => {
    expect(approachingLimit(90, 100)).toBe(true);
    expect(approachingLimit(120, 100)).toBe(true);
  });

  it("does not flag a measurement comfortably below the limit", () => {
    expect(approachingLimit(50, 100)).toBe(false);
  });

  it("fails LOUD, not silent, when the limit is unusable", () => {
    // `x >= 0.9 * NaN` is false — so a cleared config field used to switch the
    // pre-alarm cue OFF while the supply was still climbing. An unusable limit
    // must warn, never reassure.
    expect(approachingLimit(50, Number.NaN)).toBe(true);
    expect(approachingLimit(50, undefined)).toBe(true);
    expect(approachingLimit(50, Number.POSITIVE_INFINITY)).toBe(true);
  });

  it("reports no warning when there is no measurement yet", () => {
    expect(approachingLimit(undefined, 100)).toBe(false);
  });
});

describe("parseConfigNumber (#4042b)", () => {
  it("accepts a finite number", () => {
    expect(parseConfigNumber("12.5")).toBe(12.5);
    expect(parseConfigNumber("0")).toBe(0);
  });

  it("rejects input that would poison the draft with NaN/Infinity", () => {
    expect(parseConfigNumber("")).toBeNull();
    expect(parseConfigNumber("   ")).toBeNull();
    expect(parseConfigNumber("abc")).toBeNull();
    expect(parseConfigNumber("Infinity")).toBeNull();
  });
});
