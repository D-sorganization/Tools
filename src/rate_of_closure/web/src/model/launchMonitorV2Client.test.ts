import { describe, expect, it, vi } from "vitest";
import golden from "../../../launch_monitor_canonical_v2_golden.json";
import {
  buildDatasetJobRequest,
  buildPlayerCovariationPayload,
  createCanonicalLaunchMonitorClient,
  parseCanonicalDatasetReference,
  validateDatasetJobPage,
  validateLaunchMonitorStrokesGainedResponse,
  validateLaunchMonitorV2Response,
  validatePlayerCovariationResponse,
} from "./launchMonitorV2Client";

const response = { contract_version: "2.0.0", status: "available", analysis: {}, units: {},
  lineage: { dataset_fingerprint_sha256: "a".repeat(64), backing_records: [] },
  missingness: {}, availability: [], uncertainty: {}, player_identity: {}, vendor_provenance: [],
  claims: { vendor_comparison: "descriptive", device_emulation: false, device_certification: false, causal_inference: false }, warnings: [] };

describe("Upstream v2 client", () => {
  it("validates the canonical envelope and reports residuals unavailable", () => {
    expect(validateLaunchMonitorV2Response(response).rowAlignedResiduals).toMatchObject({ state: "unavailable" });
  });
  it("rejects unsafe claims", () => {
    expect(() => validateLaunchMonitorV2Response({ ...response, claims: { ...response.claims, device_emulation: true } }))
      .toThrow(/emulation/i);
  });
});

describe("canonical dataset jobs and player covariation", () => {
  it("shares the immutable dataset-reference golden with Python", () => {
    const reference = parseCanonicalDatasetReference(golden.dataset_reference);
    expect(buildDatasetJobRequest(reference, "source_summary")).toEqual(golden.dataset_job_request);
    expect(() => parseCanonicalDatasetReference({ ...golden.dataset_reference, root_id: "../private" })).toThrow(/root_id/i);
    expect(() => buildDatasetJobRequest(reference, "metric_summary", ["bogus"])).toThrow(/canonical dataset metrics/i);
  });

  it("rejects row-like dataset-job pages and oversized inline covariation", () => {
    expect(() => validateDatasetJobPage({
      contract_version: "launch-monitor-dataset-job/1.0.0", job_id: "a".repeat(32),
      offset: 0, limit: 100, total_items: 1, next_offset: null,
      items: [{ shot_id: "secret", ball_speed: 170 }],
    })).toThrow(/private rows/i);
    expect(() => validateDatasetJobPage({
      contract_version: "launch-monitor-dataset-job/1.0.0", job_id: "a".repeat(32),
      offset: 0, limit: 100, total_items: 1, next_offset: null,
      items: [{ ball_speed: 170 }],
    })).toThrow(/aggregate schema/i);
    expect(() => buildPlayerCovariationPayload(Array.from({ length: 20_001 }, () => ({})), {
      playerColumn: "player_id", xColumn: "face_angle", yColumn: "club_path",
      minSamples: 4, confidenceLevel: 0.95,
    })).toThrow(/20,000/);
  });

  it("validates safe evidence-bearing covariation responses", () => {
    const result = {
      contract_version: "launch-monitor-player-covariation/1.0.0", analysis_kind: "selected_pair",
      status: "available", request: {}, pooled: {}, within_player: {}, between_player: {},
      per_player: [], meta_analysis: {}, missingness: {}, units: {}, lineage: { backing_records: [] },
      availability: [], uncertainty: {}, player_identity: { trust_level: "explicit_user_attested", identifier_column: "player_id" },
      vendor_provenance: [], claims: { device_emulation: false, device_certification: false, causal_inference: false }, definitions: {}, warnings: [],
    };
    expect(validatePlayerCovariationResponse(result).status).toBe("available");
    expect(() => validatePlayerCovariationResponse({ ...result, claims: { ...result.claims, causal_inference: true } })).toThrow(/claim/i);
  });

  it("calls the canonical immutable dataset-job routes", async () => {
    const status = { contract_version: "launch-monitor-dataset-job/1.0.0", job_id: "a".repeat(32),
      status: "queued", submitted_at_utc: "2026-08-21T00:00:00Z", completed_at_utc: null,
      input_row_count: 0, result_item_count: 0, unavailable: null };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => status });
    vi.stubGlobal("fetch", fetchMock);
    const client = createCanonicalLaunchMonitorClient("https://authority.example/");
    await client.submitDatasetJob({ contract_version: "test" });
    await client.datasetJobStatus("a".repeat(32));
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "https://authority.example/tools/launch-monitor-analytics/v2/dataset-jobs",
      `https://authority.example/tools/launch-monitor-analytics/v2/dataset-jobs/${"a".repeat(32)}`,
    ]);
    vi.unstubAllGlobals();
  });

  it("rejects filesystem and custom-scheme authorities", () => {
    expect(() => createCanonicalLaunchMonitorClient("file:///private/corpus")).toThrow(/HTTP/i);
    expect(() => createCanonicalLaunchMonitorClient("not-a-url")).toThrow(/HTTP/i);
  });
});

describe("Upstream source-backed strokes-gained client", () => {
  it("accepts only the canonical scoring contract and safe claims", () => {
    const scoring = {
      contract_version: "launch-monitor-strokes-gained-analysis/1.0.0", status: "available",
      metric_name: "source_backed_strokes_gained", unit: "strokes",
      value_summary: { count: 3, mean: 0.25 }, baseline: { baseline_id: "test" },
      formula: "SG", units: {}, availability: {}, uncertainty: {}, row_results: [],
      excluded_rows: [], exclusions: {}, group_summaries: [], longitudinal_summaries: [],
      analysis_context: {}, dataset_fingerprint_sha256: "b".repeat(64), warnings: [], limitations: [],
      claims: { is_strokes_gained: true, source_backed: true, device_emulation: false, device_certification: false, causal_inference: false },
    };
    expect(validateLaunchMonitorStrokesGainedResponse(scoring).mean).toBe(0.25);
    expect(() => validateLaunchMonitorStrokesGainedResponse({ ...scoring, claims: { ...scoring.claims, source_backed: false } })).toThrow(/source-backed/i);
  });
});
