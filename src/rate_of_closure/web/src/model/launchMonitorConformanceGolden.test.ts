import { describe, expect, it } from "vitest";

import bundleFixture from "./__fixtures__/launch_monitor_conformance_bundle_golden_v1.json";
import covariationFixture from "./__fixtures__/launch_monitor_player_covariation_golden_v1.json";
import {
  validateLaunchMonitorStrokesGainedResponse,
  validateLaunchMonitorV2Response,
  validatePlayerCovariationResponse,
} from "./launchMonitorV2Client";

describe("launch monitor conformance bundle golden parity", () => {
  it("verifies the bundle structure and ensures no private rows are embedded", () => {
    expect(bundleFixture.bundle_version).toBe("launch-monitor-analytics-conformance/1.0.0");
    expect(bundleFixture.data_classification).toBe("synthetic_contract_fixture_no_private_rows");
    expect(bundleFixture.input_records_embedded).toBe(false);
    expect(bundleFixture.bundle_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(bundleFixture.scenarios).toHaveLength(10);

    const scenarioKinds = new Set(
      bundleFixture.scenarios.map((s) => `${s.analysis_kind}:${s.expected_status}`),
    );
    expect(scenarioKinds).toEqual(new Set([
      "analysis_v2:available",
      "analysis_v2:unavailable",
      "player_covariation:available",
      "player_covariation:unavailable",
      "attested_longitudinal:available",
      "attested_longitudinal:unavailable",
      "source_backed_strokes_gained:available",
      "source_backed_strokes_gained:unavailable",
      "distance_target_proxy:available",
      "distance_target_proxy:unavailable",
    ]));

    const jsonString = JSON.stringify(bundleFixture);
    expect(jsonString).not.toContain("restricted_internal");
  });

  it("validates safety claims, units, and backing lineage across scenarios", () => {
    for (const scenario of bundleFixture.scenarios) {
      expect(scenario.units).toBeDefined();
      expect(scenario.claims.causal_inference).toBe(false);
      if ("device_emulation" in scenario.claims) {
        expect(scenario.claims.device_emulation).toBe(false);
      }
      if ("device_certification" in scenario.claims) {
        expect(scenario.claims.device_certification).toBe(false);
      }
      expect(scenario.sources.length).toBeGreaterThan(0);
      expect(scenario.backing_records.length).toBeGreaterThan(0);

      const sourceIds = new Set(scenario.sources.map((s) => s.source_id));
      for (const record of scenario.backing_records) {
        expect(sourceIds.has(record.source_id)).toBe(true);
        expect(record.record_sha256).toMatch(/^[0-9a-f]{64}$/);
      }
    }
  });

  it("validates golden scenarios through TypeScript client response validators", () => {
    const scenariosById = new Map(
      bundleFixture.scenarios.map((s) => [s.scenario_id, s]),
    );

    const analysisAvail = scenariosById.get("analysis-v2-available")!;
    const v2Result = validateLaunchMonitorV2Response(analysisAvail.payload);
    expect(v2Result.contractVersion).toBe("2.0.0");
    expect(v2Result.payload.status).toBe("available");

    const covAvail = scenariosById.get("player-covariation-available")!;
    const covResult = validatePlayerCovariationResponse(covAvail.payload);
    expect(covResult.contract_version).toBe("launch-monitor-player-covariation/1.0.0");
    expect(covResult.status).toBe("available");

    const sgAvail = scenariosById.get("source-backed-strokes-gained-available")!;
    const sgResult = validateLaunchMonitorStrokesGainedResponse(sgAvail.payload);
    expect(sgResult.status).toBe("available");
    expect(sgResult.count).toBe(3);

    const proxyAvail = scenariosById.get("distance-target-proxy-available")!;
    expect(proxyAvail.claims.is_strokes_gained).toBe(false);
    expect(proxyAvail.claims.source_backed).toBe(false);
  });

  it("validates player covariation golden fixture parity", () => {
    expect(covariationFixture.fixture_version).toBe("launch-monitor-player-covariation-golden/1.0.0");
    expect(covariationFixture.records).toHaveLength(10);
    expect(covariationFixture.context.player_identity.trust_level).toBe("explicit_user_attested");

    const validated = validatePlayerCovariationResponse(covariationFixture.expected_result);
    expect(validated.contract_version).toBe("launch-monitor-player-covariation/1.0.0");
    expect(validated.status).toBe("available");
    const claims = validated.claims as Record<string, unknown>;
    expect(claims.causal_inference).toBe(false);
    expect(claims.device_emulation).toBe(false);

    const validatedScan = validatePlayerCovariationResponse(covariationFixture.expected_scan_result);
    expect(validatedScan.analysis_kind).toBe("pair_scan");
    expect(validatedScan.status).toBe("available");
  });
});
