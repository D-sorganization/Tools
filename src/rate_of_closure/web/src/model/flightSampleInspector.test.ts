import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/flight_sample_inspector_golden_v1.json";
import {
  flightSampleSource, navigateFlightSamples, nearestFlightSample, planFlightSamples,
} from "./flightSampleInspector";

describe("flight sample inspector", () => {
  it("matches the Python-owned phase, navigation, and pixel-tie golden", () => {
    expect(Object.keys(fixture).sort()).toEqual(["expected", "schema_id", "schema_version", "series"]);
    expect(fixture.schema_id).toBe("rate-of-closure/flight-sample-inspector-golden");
    expect(fixture.schema_version).toBe(1);
    const plan = planFlightSamples({
      timesS: fixture.series.times_s,
      positionsM: fixture.series.positions_m as [number, number, number][],
    });
    expect(plan.apexRawIndex).toBe(fixture.expected.apex_raw_index);
    expect(plan.samples.map((sample) => sample.phase)).toEqual(fixture.expected.phases);
    Object.entries(fixture.expected.navigation_from_3).forEach(([command, index]) => {
      expect(navigateFlightSamples(plan, 3, command as "next")).toBe(index);
    });
    const nearest = fixture.expected.nearest;
    expect(nearestFlightSample(
      plan,
      nearest.projected as ["current", number, number, number][], nearest.pointer as [number, number],
      nearest.hit_radius_px,
    )).toEqual({ cohort: "current", rawIndex: nearest.selection.raw_index });
  });

  it("deep-snapshots evidence and rejects malformed or oversized inputs", () => {
    const times = [0, 0.1];
    const positions: [number, number, number][] = [[0, 0, 0], [1, 0, 0]];
    const plan = planFlightSamples({ timesS: times, positionsM: positions });
    times[1] = 9;
    positions[1][0] = 9;
    expect(plan.rawSample(1)).toMatchObject({ timeS: 0.1, downrangeM: 1 });
    expect(() => planFlightSamples({ timesS: [0, 0], positionsM: [[0, 0, 0], [1, 0, 0]] })).toThrow();
    expect(() => planFlightSamples({
      timesS: Array.from({ length: 1003 }, (_, index) => index),
      positionsM: Array.from({ length: 1003 }, () => [0, 0, 0] as const),
    })).toThrow();
    expect(planFlightSamples({
      timesS: Array.from({ length: 1002 }, (_, index) => index * 0.001),
      positionsM: Array.from({ length: 1002 }, () => [0, 0, 0] as const),
    }).rawCount).toBe(1002);
    expect(() => planFlightSamples({
      timesS: new Proxy(new Array(1003), {
        get(target, property, receiver) {
          if (property !== "length") throw new Error("oversize evidence was traversed");
          return Reflect.get(target, property, receiver);
        },
      }),
      positionsM: new Array(1003),
    })).toThrow(/2\.\.1002/);
  });

  it("uses first maximum height for apex and a bounded 12 CSS-pixel hit", () => {
    const plan = planFlightSamples({
      timesS: [0, 1, 2, 3],
      positionsM: [[0, 0, 0], [1, 2, 0], [2, 2, 0], [3, 0, 0]],
    });
    expect(plan.apexRawIndex).toBe(1);
    const projected = plan.samples.map((sample) => [
      "current", sample.rawIndex, sample.rawIndex === 0 ? 0 : 100, 0,
    ] as const);
    expect(nearestFlightSample(plan, projected, [12, 0])).toEqual({ cohort: "current", rawIndex: 0 });
    expect(nearestFlightSample(plan, projected, [12.01, 0])).toBeNull();
  });

  it("rejects calm, partial, duplicate, and out-of-range projected identities", () => {
    const plan = planFlightSamples({ timesS: [0, 1], positionsM: [[0, 0, 0], [1, 0, 0]] });
    expect(() => nearestFlightSample(
      plan, [["calm" as "current", 0, 0, 0], ["calm" as "current", 1, 1, 0]], [0, 0],
    )).toThrow();
    expect(() => nearestFlightSample(plan, [["current", 0, 0, 0]], [0, 0])).toThrow();
    expect(() => nearestFlightSample(
      plan, [["current", 0, 0, 0], ["current", 0, 1, 0]], [0, 0],
    )).toThrow();
  });

  it("documents coincident launch/apex/landing precedence", () => {
    const descending = planFlightSamples({ timesS: [0, 1], positionsM: [[0, 2, 0], [1, 0, 0]] });
    const rising = planFlightSamples({ timesS: [0, 1], positionsM: [[0, 0, 0], [1, 2, 0]] });
    expect(descending.samples.map((sample) => sample.phase)).toEqual(["launch", "landing"]);
    expect(rising.samples.map((sample) => sample.phase)).toEqual(["launch", "landing"]);
    expect([descending.apexRawIndex, rising.apexRawIndex]).toEqual([0, 1]);
  });

  it("normalizes forged exploration points and snapshots public source coordinates", () => {
    expect(() => flightSampleSource({ points: [null, null] } as never)).toThrow(RangeError);
    const position: [number, number, number] = [0, 0, 0];
    const source = flightSampleSource({
      points: [
        { time: 0, position, velocity: [1, 0, 0] },
        { time: 1, position: [1, 0, 0], velocity: [1, 0, 0] },
      ],
      metrics: {} as never,
      execution: {} as never,
    });
    position[0] = 99;
    expect(source.positionsM[0][0]).toBe(0);
  });

  it.each([
    [[0, 10.001001], [[0, 0, 0], [1, 0, 0]]],
    [[0, 1], [[0, 0, 0], [10_000.001, 0, 0]]],
    [[0, 1], [[0, 0, 0], [Number.MAX_VALUE, 0, 0]]],
  ] as Array<[number[], Array<[number, number, number]>]>)(
    "rejects finite but unrenderable evidence", (timesS, positionsM) => {
    expect(() => planFlightSamples({ timesS, positionsM })).toThrow();
    },
  );
});
