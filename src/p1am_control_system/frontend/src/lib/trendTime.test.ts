import { describe, it, expect } from "vitest";
import {
  toSeconds,
  fromSeconds,
  clampWindow,
  windowSamples,
  naturalUnit,
  windowUnit,
  formatWindow,
  formatDuration,
  timeAxisTicks,
  spanSeconds,
  elapsedSeconds,
  fixedWindowRange,
  windowStartIndex,
  timeToX,
  valueToY,
  formatClock,
  timeSeriesPath,
  downsample,
  SELECTABLE_TIME_UNITS,
  MAX_WINDOW_SECONDS,
  MIN_WINDOW_SECONDS,
  BUFFER_WINDOW_SECONDS,
  MAX_TREND_SAMPLES,
  SAMPLES_PER_SECOND,
  TREND_BACKFILL_MAX_POINTS,
} from "./trendTime";

describe("toSeconds / fromSeconds", () => {
  it("converts units to seconds", () => {
    expect(toSeconds(5, "m")).toBe(300);
    expect(toSeconds(2, "h")).toBe(7200);
    expect(toSeconds(45, "s")).toBe(45);
  });

  it("validates inputs (DbC)", () => {
    expect(() => toSeconds(Number.NaN, "m")).toThrow(TypeError);
    expect(() => toSeconds(-1, "m")).toThrow(RangeError);
  });

  it("round-trips", () => {
    expect(fromSeconds(toSeconds(7, "m"), "m")).toBe(7);
  });
});

describe("window range constants (>1 hour support)", () => {
  it("lets a backfilled trend look back MORE than one hour", () => {
    expect(MAX_WINDOW_SECONDS).toBeGreaterThan(3600);
    expect(MAX_WINDOW_SECONDS).toBe(24 * 60 * 60); // 24 h
  });

  it("caps a live-buffer-only trend at exactly the buffer depth (1 h)", () => {
    expect(BUFFER_WINDOW_SECONDS).toBe(MAX_TREND_SAMPLES / SAMPLES_PER_SECOND);
    expect(BUFFER_WINDOW_SECONDS).toBe(3600);
  });

  it("decouples the backfilled max from the live-buffer depth", () => {
    // Raising the viewable window must not be tied to the in-memory sample cap.
    expect(MAX_WINDOW_SECONDS).toBeGreaterThan(BUFFER_WINDOW_SECONDS);
  });

  it("clamps a window to the (now 24 h) supported max", () => {
    expect(clampWindow(1e9)).toBe(MAX_WINDOW_SECONDS);
    expect(clampWindow(6 * 3600)).toBe(6 * 3600); // 6 h is allowed now
  });

  it("bounds the historian backfill request to a light point count", () => {
    // More than the ~600 drawn points (headroom for zooming) but nowhere near
    // the server's 200 000 cap, so a multi-hour window stays a light request.
    expect(TREND_BACKFILL_MAX_POINTS).toBeGreaterThan(600);
    expect(TREND_BACKFILL_MAX_POINTS).toBeLessThanOrEqual(200_000);
  });
});

describe("clampWindow", () => {
  it("bounds to [MIN, MAX]", () => {
    expect(clampWindow(0)).toBe(MIN_WINDOW_SECONDS);
    expect(clampWindow(1e9)).toBe(MAX_WINDOW_SECONDS);
    expect(clampWindow(120)).toBe(120);
    expect(clampWindow(Number.NaN)).toBe(MIN_WINDOW_SECONDS);
  });
});

describe("windowSamples", () => {
  it("scales by 10 Hz and floors at 2", () => {
    expect(windowSamples(30)).toBe(300);
    expect(windowSamples(0)).toBe(2);
  });
});

describe("labels", () => {
  it("picks a natural unit", () => {
    expect(naturalUnit(30)).toBe("s");
    expect(naturalUnit(90)).toBe("m");
    expect(naturalUnit(7200)).toBe("h");
  });

  it("formats windows compactly", () => {
    expect(formatWindow(30)).toBe("30s");
    expect(formatWindow(300)).toBe("5m");
    expect(formatWindow(3600)).toBe("1h");
  });

  it("formats durations as a clock", () => {
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(90)).toBe("1m30s");
    expect(formatDuration(120)).toBe("2m");
    expect(formatDuration(3600)).toBe("1h");
  });
});

describe("timeAxisTicks", () => {
  it("returns count+1 ticks from oldest to now", () => {
    const ticks = timeAxisTicks(300, 4);
    expect(ticks).toHaveLength(5);
    expect(ticks[0].frac).toBe(0);
    expect(ticks[0].label).toBe("-5m");
    expect(ticks[ticks.length - 1].frac).toBe(1);
    expect(ticks[ticks.length - 1].label).toBe("now");
  });
});

describe("spanSeconds", () => {
  it("is (n-1)/10", () => {
    expect(spanSeconds(301)).toBeCloseTo(30, 9);
    expect(spanSeconds(0)).toBe(0);
  });
});

describe("timestamp windowing (rate-independent)", () => {
  // 0.2 s spacing (5 Hz) — deliberately NOT the nominal 10 Hz.
  const times = Array.from({ length: 20 }, (_, i) => 1000 + i * 200);

  it("elapsedSeconds is exact from timestamps", () => {
    expect(elapsedSeconds(times)).toBeCloseTo((19 * 200) / 1000, 9);
    expect(elapsedSeconds([])).toBe(0);
    expect(elapsedSeconds([42])).toBe(0);
  });

  it("windowStartIndex keeps exactly the trailing window", () => {
    // last ts = 1000 + 19*200 = 4800; a 1 s window keeps ts >= 3800 -> idx 14
    expect(windowStartIndex(times, 1)).toBe(14);
    // window larger than the buffer keeps everything
    expect(windowStartIndex(times, 999)).toBe(0);
    expect(windowStartIndex([], 5)).toBe(0);
  });
});

describe("windowUnit / SELECTABLE_TIME_UNITS", () => {
  it("picks seconds / minutes / hours", () => {
    expect(windowUnit(30)).toBe("s");
    expect(windowUnit(120)).toBe("m");
    expect(windowUnit(3600)).toBe("m"); // 60 min, not 1 hr
    expect(windowUnit(7200)).toBe("h");
  });

  it("offers seconds, minutes, and hours", () => {
    expect(SELECTABLE_TIME_UNITS).toEqual(["s", "m", "h"]);
  });
});

describe("fixedWindowRange", () => {
  it("spans exactly windowSeconds ending at latestMs", () => {
    expect(fixedWindowRange(1_000_000, 60)).toEqual({ t0: 940_000, t1: 1_000_000 });
    // a 60-minute window
    expect(fixedWindowRange(0, 3600)).toEqual({ t0: -3_600_000, t1: 0 });
  });

  it("rescales immediately when the window changes (the bug being fixed)", () => {
    const latest = 5_000_000;
    expect(fixedWindowRange(latest, 30).t0).toBe(latest - 30_000);
    expect(fixedWindowRange(latest, 600).t0).toBe(latest - 600_000);
    // span is the requested window, independent of how much data exists
    const r = fixedWindowRange(latest, 120);
    expect((r.t1 - r.t0) / 1000).toBe(120);
  });
});

describe("valueToY", () => {
  // Plot area: yTop=10, plotH=100 -> [min,max] maps to [110 (bottom), 10 (top)].
  it("maps max to the top and min to the bottom", () => {
    expect(valueToY(100, 0, 100, 10, 100)).toBe(10); // max -> yTop
    expect(valueToY(0, 0, 100, 10, 100)).toBe(110); // min -> yTop + plotH
  });

  it("maps the midpoint to the middle", () => {
    expect(valueToY(50, 0, 100, 10, 100)).toBe(60);
  });

  it("clamps values outside [min,max] into the plot area", () => {
    expect(valueToY(150, 0, 100, 10, 100)).toBe(10); // above max -> top
    expect(valueToY(-20, 0, 100, 10, 100)).toBe(110); // below min -> bottom
  });

  it("returns the bottom for a degenerate range (max <= min)", () => {
    expect(valueToY(5, 10, 10, 10, 100)).toBe(110);
  });

  it("agrees with timeSeriesPath's y for the same value", () => {
    const geom = { t0: 0, t1: 10, min: 0, max: 100, x0: 0, x1: 100, yTop: 10, plotH: 100 };
    const d = timeSeriesPath(
      [
        { t: 0, v: 25 },
        { t: 10, v: 75 },
      ],
      geom,
    );
    // Second point value 75 -> valueToY(75,...) must appear as the L y-coord.
    const y = valueToY(75, 0, 100, 10, 100);
    expect(d).toContain(`,${y.toFixed(1)}`);
  });
});

describe("formatClock", () => {
  it("formats an epoch-ms timestamp as zero-padded HH:MM:SS", () => {
    // Build the expected string from the same local-time fields to stay tz-safe.
    const ms = 1_700_000_123_000;
    const d = new Date(ms);
    const pad = (n: number) => String(n).padStart(2, "0");
    expect(formatClock(ms)).toBe(
      `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`,
    );
  });

  it("zero-pads single-digit fields", () => {
    expect(formatClock(0)).toMatch(/^\d{2}:\d{2}:\d{2}$/);
  });
});

describe("timeToX", () => {
  it("maps endpoints and midpoint of the time range", () => {
    expect(timeToX(100, 100, 200, 0, 600)).toBe(0);
    expect(timeToX(200, 100, 200, 0, 600)).toBe(600);
    expect(timeToX(150, 100, 200, 0, 600)).toBe(300);
  });

  it("clamps timestamps outside the range", () => {
    expect(timeToX(50, 100, 200, 0, 600)).toBe(0);
    expect(timeToX(250, 100, 200, 0, 600)).toBe(600);
  });

  it("returns x0 for a degenerate range", () => {
    expect(timeToX(150, 100, 100, 0, 600)).toBe(0);
  });
});

describe("timeSeriesPath", () => {
  const geom = {
    t0: 0,
    t1: 100,
    min: 0,
    max: 100,
    x0: 0,
    x1: 600,
    yTop: 0,
    plotH: 200,
  };
  const xs = (d: string): number[] =>
    (d.match(/[ML]([\d.]+),/g) ?? []).map((m) => Number.parseFloat(m.slice(1)));

  it("returns empty for <2 points or a degenerate range (DbC)", () => {
    expect(timeSeriesPath([{ t: 0, v: 0 }], geom)).toBe("");
    expect(timeSeriesPath([{ t: 0, v: 0 }, { t: 1, v: 1 }], { ...geom, t1: 0 })).toBe("");
    expect(timeSeriesPath([{ t: 0, v: 0 }, { t: 1, v: 1 }], { ...geom, max: 0 })).toBe("");
  });

  it("positions points by TIME, not index (the bug being fixed)", () => {
    // The middle point is at 90% of the time span. Index-based plotting would
    // place it at x=300 (50%); time-based places it at x=540 (90%).
    const d = timeSeriesPath(
      [
        { t: 0, v: 0 },
        { t: 90, v: 50 },
        { t: 100, v: 100 },
      ],
      geom,
    );
    expect(xs(d)).toEqual([0, 540, 600]);
  });

  it("maps value to Y inverted (max at top)", () => {
    const d = timeSeriesPath(
      [
        { t: 0, v: 100 },
        { t: 100, v: 0 },
      ],
      geom,
    );
    expect(d).toBe("M0.0,0.0 L600.0,200.0");
  });
});

describe("downsample", () => {
  it("leaves short series untouched", () => {
    const v = [1, 2, 3];
    expect(downsample(v, 10)).toBe(v);
  });

  it("reduces to maxPoints and keeps the last sample", () => {
    const v = Array.from({ length: 1000 }, (_, i) => i);
    const out = downsample(v, 100);
    expect(out).toHaveLength(100);
    expect(out[out.length - 1]).toBe(999);
    expect(out[0]).toBe(0);
  });

  it("works on generic rows", () => {
    const rows = Array.from({ length: 50 }, (_, i) => [i, i * 2]);
    const out = downsample(rows, 10);
    expect(out).toHaveLength(10);
    expect(out[out.length - 1]).toEqual([49, 98]);
  });
});
