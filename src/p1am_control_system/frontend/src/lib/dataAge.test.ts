import { describe, it, expect } from "vitest";
import {
  DATA_SOURCES,
  DATA_STALE_MS,
  dataFreshness,
  dataStatus,
  formatDataAge,
  type DataFreshness,
} from "./dataAge";

/**
 * Data-age contract (#4010).
 *
 * The HMI used to express liveness as a single boolean that was latched true by
 * ANY successfully-parsed payload — including `{}` — so a dead backend rendered
 * as a rock-steady process. Liveness is now an AGE, and these lock the mapping
 * from age to the three states the operator is shown.
 */

describe("dataFreshness", () => {
  it("reports 'offline' when no frame has ever arrived", () => {
    const level: DataFreshness = dataFreshness(undefined);
    expect(level).toBe("offline");
  });

  it("reports 'live' for an age inside the stale threshold", () => {
    expect(dataFreshness(0)).toBe("live");
    expect(dataFreshness(DATA_STALE_MS - 1)).toBe("live");
  });

  it("reports 'stale' once the age reaches the threshold", () => {
    expect(dataFreshness(DATA_STALE_MS)).toBe("stale");
    expect(dataFreshness(DATA_STALE_MS * 100)).toBe("stale");
  });

  it("treats a non-finite age as offline rather than silently live", () => {
    expect(dataFreshness(Number.NaN)).toBe("offline");
    expect(dataFreshness(Number.POSITIVE_INFINITY)).toBe("offline");
  });

  it("rejects a negative age (DbC — ages cannot run backwards)", () => {
    expect(() => dataFreshness(-1)).toThrow(RangeError);
  });
});

describe("formatDataAge", () => {
  it("renders a dash when no frame has ever arrived", () => {
    expect(formatDataAge(undefined)).toBe("—");
  });

  it("renders sub-minute ages in whole seconds", () => {
    expect(formatDataAge(0)).toBe("0 s");
    expect(formatDataAge(1400)).toBe("1 s");
    expect(formatDataAge(59_900)).toBe("59 s");
  });

  it("renders minutes past a minute so a 20-minute freeze is unmistakable", () => {
    expect(formatDataAge(60_000)).toBe("1 m 0 s");
    expect(formatDataAge(20 * 60_000 + 5_000)).toBe("20 m 5 s");
  });
});

describe("dataStatus (#5525)", () => {
  it("is live only for a fresh frame from a live source", () => {
    expect(dataStatus("live", "live")).toBe("live");
  });

  it.each(["held", "fault", "simulated"] as const)(
    "a fresh %s scan is reported as %s, never live",
    (source) => {
      expect(dataStatus("live", source)).toBe(source);
    },
  );

  it.each([undefined, "", "LIVE", "garbage", "null"])(
    "fails safe to unknown for source %j on a fresh frame",
    (source) => {
      expect(dataStatus("live", source)).toBe("unknown");
    },
  );

  it("lets staleness and offline win over any reported source", () => {
    for (const source of [...DATA_SOURCES, undefined, "garbage"]) {
      expect(dataStatus("stale", source)).toBe("stale");
      expect(dataStatus("offline", source)).toBe("offline");
    }
  });
});
