import { describe, expect, it } from "vitest";

import groundFixture from "./__fixtures__/flight_to_ground_golden_v1.json";
import regionalFixture from "./__fixtures__/ground_regional_execution_golden_v1.json";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import { parseGroundRegionalExecutionResult } from "./groundRegionalExecution";
import {
  GROUND_PLAYBACK_VISUAL_POINT_BUDGET,
  GROUND_PLAYBACK_VISIBLE_ROW_BUDGET,
  GroundPlaybackTimeline,
  adjacentPlaybackTime,
  groundEvidenceWindow,
  lowerPlaybackIndex,
  selectGroundPlaybackPath,
  timelineFromRegionalExecution,
} from "./groundPlayback";

describe("GroundPlaybackTimeline", () => {
  const timeline = () =>
    new GroundPlaybackTimeline(
      parseFlightToGroundResultRecord(groundFixture.result),
    );

  it("uses absolute time, exact steps, locked endpoints and phase-safe frames", () => {
    const value = timeline();
    expect(value.startTimeS).toBeGreaterThan(0);
    expect(value.durationS).toBeGreaterThan(0);
    expect(value.carryPositionM).toEqual(value.result.trajectory[0].position_m);
    expect(value.endpointPositionM).toEqual(
      value.result.trajectory[value.result.trajectory.length - 1].position_m,
    );
    expect(value.stepTime(value.startTimeS, -1)).toBe(value.startTimeS);
    expect(value.stepTime(value.endTimeS, 1)).toBe(value.endTimeS);
    const roll = value.phaseTime("roll");
    expect(roll).not.toBeNull();
    const before = value.frameAt((roll as number) - 1e-9);
    const at = value.frameAt(roll as number);
    expect(before.phase).not.toBe(at.phase);
    expect(before.interpolationFraction).toBe(0);
  });

  it("adapts a validated regional envelope without executing physics", () => {
    const execution = parseGroundRegionalExecutionResult(
      regionalFixture.representable.result,
    );
    expect(timelineFromRegionalExecution(execution).result).toBe(
      execution.ground_result,
    );
  });

  it.each(["cancelled", "failed"] as const)(
    "rejects %s regional execution without a playable nested result",
    (name) => {
      const execution = parseGroundRegionalExecutionResult(
        regionalFixture[name].result,
      );
      expect(() => timelineFromRegionalExecution(execution)).toThrow(
        "playable ground result",
      );
    },
  );

  it("uses logarithmic lower-index lookup for a 100,000-sample timeline", () => {
    const values = Array.from({ length: 100_000 }, (_, index) => index * 0.001);
    let reads = 0;
    const observed = new Proxy(values, {
      get(target, property, receiver) {
        if (typeof property === "string" && /^\d+$/.test(property)) reads += 1;
        return Reflect.get(target, property, receiver);
      },
    });
    expect(lowerPlaybackIndex(observed, 73.456)).toBe(73_456);
    expect(reads).toBeLessThanOrEqual(20);
  });

  it("steps exact and between-sample times with logarithmic indexed reads", () => {
    const values = Array.from({ length: 100_000 }, (_, index) => index * 0.001);
    let reads = 0;
    const observed = new Proxy(values, {
      get(target, property, receiver) {
        if (typeof property === "string" && /^\d+$/.test(property)) reads += 1;
        return Reflect.get(target, property, receiver);
      },
    });
    expect(adjacentPlaybackTime(observed, values[73_456], 1)).toBe(
      values[73_457],
    );
    expect(adjacentPlaybackTime(observed, values[73_456], -1)).toBe(
      values[73_455],
    );
    expect(adjacentPlaybackTime(observed, 73.4565, 1)).toBe(values[73_457]);
    expect(adjacentPlaybackTime(observed, 73.4565, -1)).toBe(values[73_456]);
    expect(adjacentPlaybackTime(observed, values[0], -1)).toBe(values[0]);
    expect(adjacentPlaybackTime(observed, values[values.length - 1], 1)).toBe(
      values[values.length - 1],
    );
    expect(reads).toBeLessThanOrEqual(140);
  });

  it("bounds visual points while retaining endpoints and phase/event evidence", () => {
    const base = parseFlightToGroundResultRecord(groundFixture.result);
    const trajectory = Array.from({ length: 100_000 }, (_, index) => ({
      ...base.trajectory[0],
      time_s: index * 0.001,
      phase:
        index < 10
          ? ("impact" as const)
          : index < 50_000
            ? ("skid" as const)
            : ("roll" as const),
      position_m: [index * 0.001, 0, 0] as const,
    }));
    const result = {
      ...base,
      trajectory,
      events: [
        {
          ...base.events[0],
          time_s: 25,
          position_m: trajectory[25_000].position_m,
        },
        {
          ...base.events[0],
          sequence: 1,
          time_s: 75,
          position_m: trajectory[75_000].position_m,
        },
      ],
    };
    const visible = selectGroundPlaybackPath(result);
    const times = new Set(visible.map(({ time_s }) => time_s));
    expect(visible.length).toBeLessThanOrEqual(
      GROUND_PLAYBACK_VISUAL_POINT_BUDGET,
    );
    expect([...times]).toEqual(
      expect.arrayContaining(
        [0, 9, 10, 25_000, 49_999, 50_000, 75_000, 99_999].map(
          (index) => trajectory[index].time_s,
        ),
      ),
    );
  });

  it("exposes only the bounded evidence window with exact disclosure", () => {
    const source = Array.from({ length: 100_000 }, (_, index) => index);
    const window = groundEvidenceWindow(source);
    expect(window.rows).toHaveLength(GROUND_PLAYBACK_VISIBLE_ROW_BUDGET);
    expect(window.rows[0]).toBe(0);
    expect(window.rows[255]).toBe(255);
    expect(window.disclosure).toBe(
      "Showing first 256 of 100000 validated rows; full result retained.",
    );
  });
});

describe("workspace speed whitelist vs the shared playback transport", () => {
  // SUPPORTED_PLAYBACK_SPEEDS deliberately stays its own constant: it
  // validates a versioned, fail-closed persisted document, so what it
  // accepts is a wire contract that must not silently follow a runtime
  // refactor. PLAYBACK_SPEEDS is what every playback surface (Qt
  // PlaybackTransportControls, React PlaybackTransportBar) actually offers.
  // Today the two are equal by coincidence; this gate makes the equality a
  // fact. If it fails, either the transport grew a speed the wire must learn
  // to accept (a workspace wire-version question - raise it on the
  // workspace's issue, do not just widen the whitelist) or the wire accepts
  // a speed no player offers. Twin: test_ground_playback_workspace_v2.py.
  it("keeps the two speed sets identical", async () => {
    const { SUPPORTED_PLAYBACK_SPEEDS } =
      await import("./groundPlaybackWorkspace");
    const { PLAYBACK_SPEEDS } = await import("./playbackTransport");
    expect([...SUPPORTED_PLAYBACK_SPEEDS]).toEqual([...PLAYBACK_SPEEDS]);
  });
});
