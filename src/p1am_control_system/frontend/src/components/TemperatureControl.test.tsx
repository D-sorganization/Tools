import { describe, expect, it } from "vitest";
import { recallSetpointText } from "./TemperatureControl";
import {
  formatTcReadout,
  heatUpRateReadout,
  plotPxToTime,
} from "../lib/temperatureTrend";

/**
 * Unit tests for the pure setpoint-recall helper. Rendering the full
 * TemperatureControl pulls in the WebSocket-fed trend and the config fetch,
 * so the pre-fill decision is extracted into recallSetpointText() and tested
 * directly here (light + green, per the task).
 */
describe("recallSetpointText", () => {
  it("pre-fills the recalled last setpoint when untouched", () => {
    expect(recallSetpointText(250, false)).toBe("250.0");
    expect(recallSetpointText(37.25, false)).toBe("37.3");
    expect(recallSetpointText(0, false)).toBe("0.0");
  });

  it("does not overwrite once the operator has typed this session", () => {
    expect(recallSetpointText(250, true)).toBeNull();
  });

  it("returns null for a missing or non-finite recalled value", () => {
    expect(recallSetpointText(null, false)).toBeNull();
    expect(recallSetpointText(undefined, false)).toBeNull();
    expect(recallSetpointText(Number.NaN, false)).toBeNull();
    expect(recallSetpointText(Infinity, false)).toBeNull();
  });

  it("guards its precondition (DbC)", () => {
    // @ts-expect-error deliberately wrong type for the guard test
    expect(() => recallSetpointText(100, "yes")).toThrow(TypeError);
  });
});

/**
 * Live thermocouple readout formatting for the selector. Pure so the operator's
 * "spot a dead sensor at a glance" requirement can be verified without a render.
 */
describe("formatTcReadout", () => {
  it("formats a finite reading to one decimal with a unit", () => {
    expect(formatTcReadout(123.45)).toBe("123.5 °C");
    expect(formatTcReadout(0)).toBe("0.0 °C");
    expect(formatTcReadout(-5.2)).toBe("-5.2 °C");
  });

  it("shows an em-dash for a missing or non-finite reading", () => {
    expect(formatTcReadout(null)).toBe("—");
    expect(formatTcReadout(undefined)).toBe("—");
    expect(formatTcReadout(Number.NaN)).toBe("—");
    expect(formatTcReadout(Infinity)).toBe("—");
  });
});

/**
 * Heat-up-rate readout built from the rolling buffer + operator choices. Tested
 * directly (the fit/window/rate math lives in the shared curveFit lib) so the
 * component only wires it up.
 */
describe("heatUpRateReadout", () => {
  // A perfectly linear ramp of the active channel: +1 °C per 60 s = +1 °C/min.
  const t0 = 1_000_000;
  const linearSamples = Array.from({ length: 6 }, (_, i) => ({
    t: t0 + i * 60_000, // one sample per minute
    k: 100 + i, // +1 °C/min on K
    r: 200 - i, // R present but unused when K is active
    relayOn: i % 2 === 0,
  }));

  it("reports the ramp rate and R² of the active channel when a fit is selected", () => {
    const out = heatUpRateReadout(linearSamples, "K", "linear", 60);
    expect(out.hasFit).toBe(true);
    expect(out.rate).toBe("+1.0 °C/min · +60 °C/hr");
    expect(out.r2).toBe("R² = 1.000");
  });

  it("fits the OTHER channel when it is the active one", () => {
    const out = heatUpRateReadout(linearSamples, "R", "linear", 60);
    expect(out.hasFit).toBe(true);
    // R falls at 1 °C/min → negative ramp.
    expect(out.rate).toBe("−1.0 °C/min · −60 °C/hr");
  });

  it("returns a neutral placeholder when no fit method is selected", () => {
    const out = heatUpRateReadout(linearSamples, "K", "none", 60);
    expect(out).toEqual({ rate: "—", r2: "", hasFit: false });
  });

  it("returns a placeholder when the window leaves too few points", () => {
    // A 0.1 min window over 1-min-spaced samples keeps only the last point.
    const out = heatUpRateReadout(linearSamples, "K", "linear", 0.1);
    expect(out.hasFit).toBe(false);
    expect(out.rate).toBe("—");
  });

  it("drops null readings so a dead-sensor gap never poisons the fit", () => {
    const withGap = [
      { t: t0, k: 100, r: null, relayOn: false },
      { t: t0 + 60_000, k: null, r: null, relayOn: false }, // dropped
      { t: t0 + 120_000, k: 102, r: null, relayOn: false },
    ];
    const out = heatUpRateReadout(withGap, "K", "linear", 60);
    expect(out.hasFit).toBe(true);
    // 2 °C over 2 min = +1 °C/min through the two finite points.
    expect(out.rate).toBe("+1.0 °C/min · +60 °C/hr");
  });

  it("guards its precondition (DbC)", () => {
    expect(() =>
      heatUpRateReadout(linearSamples, "K", "linear", Number.NaN),
    ).toThrow(TypeError);
  });
});

/**
 * Pixel→time mapping that backs the trend's wheel- and drag-to-zoom (the crux of
 * scroll-back / zoom). Pure so the interaction math is verified without a DOM;
 * the pan/zoom/pause state itself lives in the tested useTrendViewport hook.
 */
describe("plotPxToTime", () => {
  const plotW = 550; // TREND_W(600) - PAD_L(40) - PAD_R(10)

  it("maps the plot edges to the window bounds", () => {
    expect(plotPxToTime(0, plotW, 1000, 5000)).toBe(1000);
    expect(plotPxToTime(plotW, plotW, 1000, 5000)).toBe(5000);
  });

  it("linearly interpolates a pixel to a time inside the window", () => {
    // 50% across the plot → midpoint of the [1000, 5000] window.
    expect(plotPxToTime(plotW / 2, plotW, 1000, 5000)).toBe(3000);
    // 25% across → 25% of the span.
    expect(plotPxToTime(plotW * 0.25, plotW, 0, 4000)).toBe(1000);
  });

  it("degenerates to t0 for a zero-width plot (DbC)", () => {
    expect(plotPxToTime(100, 0, 1000, 5000)).toBe(1000);
  });
});
