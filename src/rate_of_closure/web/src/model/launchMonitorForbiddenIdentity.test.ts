import { describe, expect, it } from "vitest";

import goldenV3 from "./__fixtures__/launch_monitor_workspace_v3_golden.json";
import {
  buildPlayerCovariationPayload,
  validatePlayerCovariationResponse,
} from "./launchMonitorV2Client";
import {
  parseWorkspaceV3,
} from "./launchMonitorWorkspaceV3";

describe("launch monitor forbidden identity policy", () => {
  it("rejects workspace v3 projects with unattested player identity", () => {
    const unattested = {
      ...goldenV3,
      identity_evidence: {
        ...goldenV3.identity_evidence,
        player: {
          ...goldenV3.identity_evidence.player,
          user_attested: false,
        },
      },
    };
    expect(() => parseWorkspaceV3(unattested)).toThrow(/attested/i);
  });

  it("rejects workspace v3 projects with missing or empty player identity column", () => {
    const emptyColumn = {
      ...goldenV3,
      identity_evidence: {
        ...goldenV3.identity_evidence,
        player: {
          ...goldenV3.identity_evidence.player,
          column: "",
        },
      },
    };
    expect(() => parseWorkspaceV3(emptyColumn)).toThrow(/column/i);
  });

  it("rejects workspace v3 projects with missing evidence text", () => {
    const emptyEvidence = {
      ...goldenV3,
      identity_evidence: {
        ...goldenV3.identity_evidence,
        player: {
          ...goldenV3.identity_evidence.player,
          evidence: "",
        },
      },
    };
    expect(() => parseWorkspaceV3(emptyEvidence)).toThrow(/evidence/i);
  });

  it("buildPlayerCovariationPayload rejects non-distinct or empty columns", () => {
    const records = [{ player_id: "p1", face_angle: 1.0, club_path: 0.5 }];
    expect(() => buildPlayerCovariationPayload(records, {
      playerColumn: "face_angle",
      xColumn: "face_angle",
      yColumn: "club_path",
      minSamples: 4,
      confidenceLevel: 0.95,
    })).toThrow(/distinct/i);

    expect(() => buildPlayerCovariationPayload(records, {
      playerColumn: "   ",
      xColumn: "face_angle",
      yColumn: "club_path",
      minSamples: 4,
      confidenceLevel: 0.95,
    })).toThrow(/distinct/i);
  });

  it("validatePlayerCovariationResponse rejects untrusted or inferred identity trust levels", () => {
    const invalidTrust = {
      contract_version: "launch-monitor-player-covariation/1.0.0",
      analysis_kind: "selected_pair",
      status: "available",
      request: {},
      pooled: {},
      within_player: {},
      between_player: {},
      per_player: [],
      meta_analysis: {},
      missingness: {},
      units: {},
      lineage: { backing_records: [] },
      availability: [],
      uncertainty: {},
      player_identity: {
        trust_level: "inferred_from_session",
        identifier_column: "session_id",
      },
      vendor_provenance: [],
      claims: {
        device_emulation: false,
        device_certification: false,
        causal_inference: false,
      },
      definitions: {},
      warnings: [],
    };
    expect(() => validatePlayerCovariationResponse(invalidTrust)).toThrow(/trusted identity/i);
  });

  it("rejects workspace v3 projects attempting to smuggle rows in identity metadata", () => {
    const smuggled = {
      ...goldenV3,
      identity_evidence: {
        ...goldenV3.identity_evidence,
        player: {
          ...goldenV3.identity_evidence.player,
          rows: [{ player_id: "p1" }],
        },
      },
    };
    expect(() => parseWorkspaceV3(smuggled)).toThrow(/unknown|rows/i);
  });
});
