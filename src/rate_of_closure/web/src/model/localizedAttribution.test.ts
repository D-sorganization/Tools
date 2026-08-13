import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/localized_attribution_authority_v1.json";
import {
  attributionAuthorityFromValue,
  attributionAuthorityToValue,
  attributionObservationsToCsv,
  attributionObservationsToRows,
  attributionViewFromJson,
  attributionViewToJson,
  buildAttributionView,
  type AttributionViewDefinitionTs,
} from "./localizedAttribution";
import csvRows from "./__fixtures__/localized_attribution_csv_rows_v1.json";

const authority = () => attributionAuthorityFromValue(structuredClone(fixture));
const definition = (
  targetId = "state.clubhead.x.0_002",
  perturbedTrialIndex = 2,
): AttributionViewDefinitionTs => ({
  schemaId: "rate-of-closure/localized-attribution-view",
  schemaVersion: 1,
  authorityId: "fixture.localized-attribution.v1",
  sourceSpecId: "fixture.shoulder",
  targetId,
  baselineTrialIndex: 0,
  perturbedTrialIndex,
});

interface MutableFixture {
  schema_version: unknown;
  sources: Array<{ time_window_s: unknown[]; joint_id: unknown }>;
  observations: Array<{
    response: unknown;
    perturbed_target_value: unknown;
  }>;
}

describe("localized attribution authority", () => {
  it("strictly round-trips the Python-owned parity fixture", () => {
    const decoded = authority();
    expect(attributionAuthorityToValue(decoded)).toEqual(fixture);
    expect(decoded.sources[0].jointId).toBe("joint.shoulder");
    expect(decoded.targets[0].pointId).toBe("swing.clubhead.reference");
  });

  it("rejects empty authorities and forged typed writer inputs", () => {
    const empty = structuredClone(fixture);
    empty.observations = [];
    expect(() => attributionAuthorityFromValue(empty)).toThrow(/observations/);

    const forged = structuredClone(authority()) as unknown as { authorityId: unknown };
    forged.authorityId = 42;
    expect(() => attributionAuthorityToValue(
      forged as unknown as ReturnType<typeof authority>,
    )).toThrow(/authority_id/);

    const forgedView = { ...definition(), baselineTrialIndex: "0" };
    expect(() => attributionViewToJson(
      forgedView as unknown as AttributionViewDefinitionTs,
    )).toThrow(/baseline_trial_index/);
  });

  it("selects a retained pair and accounts for misses and failures", () => {
    const view = buildAttributionView(authority(), definition());
    expect(view.selected.response).toBeCloseTo(0.1);
    expect(view.selected.perturbedStatus).toBe("evaluated_no_impact");
    expect(view.denominator).toEqual({
      totalPairs: 3,
      availablePairs: 2,
      typedNoImpactPairs: 1,
      unavailableNoImpactPairs: 0,
      failedPairs: 1,
      nonfinitePairs: 0,
    });
  });

  it("never converts unavailable impact values into zero", () => {
    const view = buildAttributionView(
      authority(), definition("impact.clubhead_speed"),
    );
    expect(view.selected.perturbedTargetValue).toBeNull();
    expect(view.selected.response).toBeNull();
    expect(view.selected.availability).toBe("no_impact_unavailable");
    expect(view.denominator.unavailableNoImpactPairs).toBe(1);
  });

  it("strictly persists view selection without numeric coercion", () => {
    const encoded = attributionViewToJson(definition("shot.carry", 1));
    expect(attributionViewFromJson(encoded)).toEqual(definition("shot.carry", 1));
    const coercive = JSON.parse(encoded) as Record<string, unknown>;
    coercive.baseline_trial_index = "0";
    expect(() => attributionViewFromJson(JSON.stringify(coercive))).toThrow(
      /baseline_trial_index/,
    );
    expect(() => attributionViewFromJson(JSON.stringify({
      ...JSON.parse(encoded), extra: true,
    }))).toThrow(/fields/);
  });

  it("exports raw typed observations with the noncausal interpretation", () => {
    const csv = attributionObservationsToCsv(authority());
    expect(csv).toContain("schema_id,schema_version,authority_id,interpretation");
    expect(csv).toContain("source_variable,source_unit");
    expect(csv).toContain("target_unit,target_frame,target_convention");
    expect(csv).toContain("paired-planted-intervention-noncausal");
    expect(csv).toContain("no_impact_unavailable");
    expect(csv).toContain("numerical_failure");
    expect(csv).toContain(",-2,");
    expect(csv).not.toContain(",'-2,");
    expect(attributionObservationsToRows(authority())).toEqual(csvRows);
  });

  it("enforces the complete pair roster matrix and cross-target identity", () => {
    const missing = structuredClone(fixture);
    missing.observations.pop();
    expect(() => attributionAuthorityFromValue(missing)).toThrow(/matrix/);
    const forged = structuredClone(fixture);
    forged.observations[3].perturbed_source_value = 99;
    expect(() => attributionAuthorityFromValue(forged)).toThrow(/pair roster/);
  });

  it("enforces safe resources, canonical targets, and shared response tolerance", () => {
    const unsafe = structuredClone(fixture);
    unsafe.pairs[0].baseline_trial_index = Number.MAX_SAFE_INTEGER + 1;
    expect(() => attributionAuthorityFromValue(unsafe)).toThrow(/safe integer/);
    const target = structuredClone(fixture);
    target.targets[0].unit = "ft";
    expect(() => attributionAuthorityFromValue(target)).toThrow(/target registry/);
    const boundary = structuredClone(fixture);
    const row = boundary.observations[0];
    const expected = row.perturbed_target_value! - row.baseline_target_value!;
    row.response = expected + 4 * Number.EPSILON;
    expect(() => attributionAuthorityFromValue(boundary)).not.toThrow();
    row.response = expected + 8 * Number.EPSILON;
    expect(() => attributionAuthorityFromValue(boundary)).toThrow(/response/);
    const oversized = structuredClone(fixture) as unknown as Record<string, unknown>;
    oversized.sources = Array.from({ length: 33 });
    expect(() => attributionAuthorityFromValue(oversized)).toThrow(/resource cap/);
  });

  it("deep-freezes parsed authority and view values", () => {
    const decoded = authority();
    expect(Object.isFrozen(decoded)).toBe(true);
    expect(Object.isFrozen(decoded.sources[0].timeWindowS)).toBe(true);
    const view = attributionViewFromJson(attributionViewToJson(definition()));
    expect(Object.isFrozen(view)).toBe(true);
  });

  it.each([
    ["schema version", (raw: MutableFixture) => { raw.schema_version = "1"; }],
    ["window", (raw: MutableFixture) => {
      raw.sources[0].time_window_s[0] = "0.001";
    }],
    ["joint", (raw: MutableFixture) => { raw.sources[0].joint_id = "swing.wrist"; }],
    ["response", (raw: MutableFixture) => { raw.observations[0].response = 99; }],
    ["unavailable value", (raw: MutableFixture) => {
      raw.observations[4].perturbed_target_value = 0;
    }],
  ])("rejects a coercive or forged %s", (_name, mutate) => {
    const raw = structuredClone(fixture) as unknown as MutableFixture;
    (mutate as (value: MutableFixture) => void)(raw);
    expect(() => attributionAuthorityFromValue(raw)).toThrow();
  });
});
