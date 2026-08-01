import { describe, it, expect } from "vitest";
import {
  activeAlarmSchema,
  telemetryFrameSchema,
  hasTelemetryContent,
  partitionAlarmMap,
} from "./schemas";

/**
 * Regression tests for the telemetry frame contract.
 *
 * A strict `value: z.number()` on the alarm schema once made every ACTIVE alarm
 * fail validation, which failed the whole frame in `applyFrame`, leaving the HMI
 * permanently OFFLINE with dead controls the moment any alarm fired. These lock
 * the contract to the ACTUAL backend payload and prove one bad field can no
 * longer take down the live stream.
 *
 * The over-correction was just as dangerous (#4011): `.catch(undefined)` sat on
 * the WHOLE `active_alarms` record, so ONE malformed alarm erased every OTHER
 * alarm from the frame and the HMI kept rendering "All normal — no active
 * alarms" while the PLC was in alarm. Resilience is now per-entry.
 */

// The exact shape the backend's alarm engine emits (tag_name, NO value).
const realAlarm = {
  tag_id: "TAG_0",
  tag_name: "TAG_0",
  state: "LoLo",
  severity: 2,
  acknowledged: false,
  timestamp: "2026-07-08T16:15:17.404104+00:00",
};

describe("activeAlarmSchema", () => {
  it("accepts the real backend alarm shape (tag_name present, no value)", () => {
    expect(activeAlarmSchema.safeParse(realAlarm).success).toBe(true);
  });

  it("still accepts an alarm that does include a numeric value", () => {
    expect(activeAlarmSchema.safeParse({ ...realAlarm, value: 3.14 }).success).toBe(
      true,
    );
  });
});

describe("telemetryFrameSchema", () => {
  it("parses a frame with a populated active_alarms map (the reboot scenario)", () => {
    const frame = {
      tags: [1, 2, 3],
      tags_dict: { TAG_0: 1 },
      active_alarms: {
        TAG_0: realAlarm,
        TAG_1: { ...realAlarm, tag_id: "TAG_1" },
      },
      e_stop_active: false,
      temperature: { state: "idle", burnout_high_side: true },
    };
    const parsed = telemetryFrameSchema.safeParse(frame);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.active_alarms?.TAG_0?.tag_id).toBe("TAG_0");
      expect(parsed.data.tags).toEqual([1, 2, 3]);
    }
  });

  it("does NOT reject the whole frame when one field is malformed (resilience)", () => {
    const frame = {
      tags: [10, 20, 30],
      temperature: { state: "running" },
      active_alarms: { BAD: { nonsense: true } },
    };
    const parsed = telemetryFrameSchema.safeParse(frame);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.tags).toEqual([10, 20, 30]);
    }
  });

  it("drops ONLY the malformed alarm, never the whole alarm map (#4011)", () => {
    // This is the scenario the previous suite asserted as CORRECT: it expected
    // `active_alarms` to become undefined, which is exactly the defect — every
    // GOOD alarm vanished along with the bad one and the HMI reported normal.
    const frame = {
      tags: [10, 20, 30],
      active_alarms: {
        TAG_0: realAlarm,
        BAD: { nonsense: true },
        TAG_7: { ...realAlarm, tag_id: "TAG_7", severity: 1 },
      },
    };
    const parsed = telemetryFrameSchema.safeParse(frame);
    expect(parsed.success).toBe(true);
    if (!parsed.success) return;

    expect(parsed.data.active_alarms).toBeDefined();
    const { alarms, droppedIds } = partitionAlarmMap(parsed.data.active_alarms);
    expect(alarms.map((a) => a.tag_id).sort()).toEqual(["TAG_0", "TAG_7"]);
    expect(droppedIds).toEqual(["BAD"]);
  });

  it("keeps temperature (opaque) intact so heater controls stay live", () => {
    const parsed = telemetryFrameSchema.safeParse({
      temperature: { state: "running", setpoint_c: 1200, relay_on: true },
    });
    expect(parsed.success).toBe(true);
  });
});

describe("partitionAlarmMap", () => {
  it("returns empty results for an absent map", () => {
    expect(partitionAlarmMap(undefined)).toEqual({ alarms: [], droppedIds: [] });
  });

  it("keeps every well-formed entry", () => {
    const { alarms, droppedIds } = partitionAlarmMap({
      TAG_0: realAlarm,
      TAG_1: { ...realAlarm, tag_id: "TAG_1" },
    });
    expect(alarms).toHaveLength(2);
    expect(droppedIds).toEqual([]);
  });
});

describe("hasTelemetryContent", () => {
  it("rejects an empty object — a dead backend's `{}` is NOT a live frame (#4010)", () => {
    const parsed = telemetryFrameSchema.safeParse({});
    // An empty payload still PARSES (every field is optional) — which is why
    // liveness must be decided on content, not on parse success.
    expect(parsed.success).toBe(true);
    if (parsed.success) expect(hasTelemetryContent(parsed.data)).toBe(false);
  });

  it("rejects a payload made only of fields the HMI does not recognise", () => {
    const parsed = telemetryFrameSchema.safeParse({ unrelated: 1, other: "x" });
    expect(parsed.success).toBe(true);
    if (parsed.success) expect(hasTelemetryContent(parsed.data)).toBe(false);
  });

  it("accepts a frame carrying any single recognised field", () => {
    const cases: unknown[] = [
      { tags: [1, 2] },
      { tags_dict: { TAG_0: 1 } },
      { alicats: [] },
      { active_alarms: {} },
      { e_stop_active: false },
      { power_supply: { state: "idle" } },
      { temperature: { state: "idle" } },
    ];
    for (const raw of cases) {
      const parsed = telemetryFrameSchema.safeParse(raw);
      expect(parsed.success).toBe(true);
      if (parsed.success) expect(hasTelemetryContent(parsed.data)).toBe(true);
    }
  });

  it("counts `e_stop_active: false` as content (a real boolean, not absence)", () => {
    const parsed = telemetryFrameSchema.safeParse({ e_stop_active: false });
    expect(parsed.success).toBe(true);
    if (parsed.success) expect(hasTelemetryContent(parsed.data)).toBe(true);
  });
});
