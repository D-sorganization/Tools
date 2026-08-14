import { describe, it, expect } from "vitest";
import { activeAlarmSchema, telemetryFrameSchema } from "./schemas";

/**
 * Regression tests for the telemetry frame contract.
 *
 * A strict `value: z.number()` on the alarm schema once made every ACTIVE alarm
 * fail validation, which failed the whole frame in `applyFrame`, leaving the HMI
 * permanently OFFLINE with dead controls the moment any alarm fired. These lock
 * the contract to the ACTUAL backend payload and prove one bad field can no
 * longer take down the live stream.
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
    expect(
      activeAlarmSchema.safeParse({ ...realAlarm, value: 3.14 }).success,
    ).toBe(true);
  });
});

describe("telemetryFrameSchema", () => {
  it("parses a frame with a populated active_alarms map (the reboot scenario)", () => {
    const frame = {
      tags: [1, 2, 3],
      tags_dict: { TAG_0: 1 },
      active_alarms: { TAG_0: realAlarm, TAG_1: { ...realAlarm, tag_id: "TAG_1" } },
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
      // Garbage alarms that would fail activeAlarmSchema.
      active_alarms: { BAD: { nonsense: true } },
    };
    const parsed = telemetryFrameSchema.safeParse(frame);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      // The good fields survive; only the bad field is dropped.
      expect(parsed.data.tags).toEqual([10, 20, 30]);
      expect(parsed.data.active_alarms).toBeUndefined();
    }
  });

  it("keeps temperature (opaque) intact so heater controls stay live", () => {
    const parsed = telemetryFrameSchema.safeParse({
      temperature: { state: "running", setpoint_c: 1200, relay_on: true },
    });
    expect(parsed.success).toBe(true);
  });

  it("preserves signal quality, timing, diagnostic, source, and sequence", () => {
    const parsed = telemetryFrameSchema.safeParse({
      tag_samples: {
        TAG_0: {
          value: 12.5,
          source_timestamp: "2026-08-03T20:00:00+00:00",
          server_timestamp: "2026-08-03T20:00:01+00:00",
          quality: "stale",
          diagnostic_reason: "read_timeout",
          sequence: 42,
          source: "synthetic.driver",
        },
      },
      comms_health: {
        quality: "stale",
        diagnostic_reason: "read_timeout",
        sequence: 42,
        server_timestamp: "2026-08-03T20:00:01+00:00",
        source: "synthetic.driver",
      },
    });

    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.tag_samples?.TAG_0.quality).toBe("stale");
      expect(parsed.data.comms_health?.sequence).toBe(42);
    }
  });
});
