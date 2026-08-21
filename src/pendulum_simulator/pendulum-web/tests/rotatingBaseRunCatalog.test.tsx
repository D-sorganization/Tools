import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RotatingBaseTraceCharts } from "../src/components/RotatingBaseTraceCharts";
import {
  ROTATING_BASE_BOUNDARIES,
  ROTATING_BASE_MODEL_TIER,
  ROTATING_BASE_SOURCE_REVISION,
  ROTATING_BASE_STUDY,
  ROTATING_BASE_STUDY_SHA256,
} from "../src/rotatingBaseStudy";
import {
  ROTATING_BASE_RUN_CATALOG,
  ROTATING_BASE_RUN_CATALOG_SHA256,
  registeredRun,
  validateRotatingBaseRunCatalog,
} from "../src/rotatingBaseRunCatalog";

function qualifiedCatalog() {
  return {
    schema_id: "swing-sim/rotating-base-run-catalog",
    schema_version: 1,
    source_revision: ROTATING_BASE_SOURCE_REVISION,
    study_sha256: ROTATING_BASE_STUDY_SHA256,
    model_tier: ROTATING_BASE_MODEL_TIER,
    attempted_run_count: 18,
    runs: ROTATING_BASE_STUDY.cases.map((retainedCase) => ({
      schema_id: "swing-sim/rotating-base-run-result",
      schema_version: 1,
      source_revision: ROTATING_BASE_SOURCE_REVISION,
      model_tier: ROTATING_BASE_MODEL_TIER,
      boundaries: ROTATING_BASE_BOUNDARIES,
      request: {
        case_index: retainedCase.case_index,
        torso_profile: retainedCase.torso_profile,
        matching_rule: retainedCase.matching_rule,
        initial_torso_rate_rad_s: retainedCase.initial_torso_rate_rad_s,
      },
      case: {
        case_index: retainedCase.case_index,
        torso_profile: retainedCase.torso_profile,
        matching_rule: retainedCase.matching_rule,
        initial_torso_rate_rad_s: retainedCase.initial_torso_rate_rad_s,
        valid: retainedCase.valid,
        exclusion_reasons: retainedCase.exclusion_reasons,
        metrics: Object.fromEntries(
          Object.entries(retainedCase).filter(
            ([key]) =>
              ![
                "case_index",
                "torso_profile",
                "matching_rule",
                "initial_torso_rate_rad_s",
                "valid",
                "exclusion_reasons",
              ].includes(key),
          ),
        ),
      },
      trace: {
        time_s: [0, 0.0005],
        torso_rate_rad_s: [1, 2],
        club_rate_rad_s: [3, 4],
        clubhead_speed_m_s: [5, 6],
        contact_power_on_club_w: [7, 8],
        force_generated_couple_nm: [9, 10],
        force_on_club_n: [
          [[3, 4], [5, 12]],
          [[8, 15], [7, 24]],
        ],
        distal_segment_kinetic_energy_j: [11, 12],
      },
    })),
  };
}

describe("qualified rotating-base run catalog", () => {
  it("pins and validates the generated full-resolution authority", () => {
    const resource = readFileSync(
      new URL(
        "../../../shared/python/swing_sim/rotating_base/resources/rotating_base_registered_runs_v1.json",
        import.meta.url,
      ),
      "utf8",
    ).trimEnd();

    expect(createHash("sha256").update(resource).digest("hex")).toBe(
      ROTATING_BASE_RUN_CATALOG_SHA256,
    );
    expect(ROTATING_BASE_RUN_CATALOG.runs).toHaveLength(18);
    expect(ROTATING_BASE_RUN_CATALOG.runs[0].trace.time_s.length).toBeGreaterThan(200);
    expect(
      ROTATING_BASE_RUN_CATALOG.runs
        .filter((run) => !run.case.valid)
        .map((run) => run.request.case_index),
    ).toEqual([6, 7, 8, 15, 16]);
  });

  it("retains all full-resolution registered run identities", () => {
    const catalog = validateRotatingBaseRunCatalog(qualifiedCatalog());
    const adverse = registeredRun(catalog, "decelerate", "absolute_club_rate", 3.5);

    expect(catalog.runs).toHaveLength(18);
    expect(adverse.request.case_index).toBe(16);
    expect(adverse.case.valid).toBe(false);
    expect(adverse.trace.force_on_club_n[0]).toHaveLength(2);
  });

  it("fails closed on reordered, scalar-tampered, or malformed trace evidence", () => {
    const reordered = structuredClone(qualifiedCatalog());
    [reordered.runs[0], reordered.runs[1]] = [reordered.runs[1], reordered.runs[0]];
    expect(() => validateRotatingBaseRunCatalog(reordered)).toThrow(
      "does not match the qualified study row",
    );

    const scalarTampered = structuredClone(qualifiedCatalog());
    scalarTampered.runs[0].case.metrics.impact_speed_m_s += 1e-4;
    expect(() => validateRotatingBaseRunCatalog(scalarTampered)).toThrow(
      "does not match the qualified study",
    );

    const malformedTrace = structuredClone(qualifiedCatalog());
    malformedTrace.runs[0].trace.force_on_club_n[0].pop();
    expect(() => validateRotatingBaseRunCatalog(malformedTrace)).toThrow(
      "must retain two hands",
    );
  });

  it("renders every reviewer trace channel with bilateral grip separation", () => {
    const catalog = validateRotatingBaseRunCatalog(qualifiedCatalog());
    const html = renderToString(<RotatingBaseTraceCharts run={catalog.runs[0]} />);

    expect(html).toContain("Time-Resolved Registered Evidence");
    expect(html).toContain("Club Contact Power");
    expect(html).toContain("Force-Generated Couple");
    expect(html).toContain("Torso and Club Rates");
    expect(html).toContain("Distal Segment Kinetic Energy");
    expect(html).toContain("Independent Grip-Force Magnitudes");
  });
});
