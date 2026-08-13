import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/localized_attribution_authority_v1.json";
import {
  attributionAuthorityFromValue,
  attributionAuthorityToValue,
  attributionObservationsToCsv,
  attributionViewFromJson,
  attributionViewToJson,
  buildAttributionView,
  type AttributionViewDefinitionTs,
} from "./localizedAttribution";

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

    const forged = authority() as unknown as { authorityId: unknown };
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
    expect(csv).toContain("interpretation,source_spec_id,joint_id,window_start_s");
    expect(csv).toContain("paired-planted-intervention-noncausal");
    expect(csv).toContain("no_impact_unavailable");
    expect(csv).toContain("numerical_failure");
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
