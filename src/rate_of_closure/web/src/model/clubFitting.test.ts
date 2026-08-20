import { describe, expect, it } from "vitest";
import {
  parseBodyChain,
  parseFittingDocument,
  parseFittingReport,
  parseImpactCouplingReport,
  gripBoundaryReduction,
  FITTING_DOCUMENT_FORMAT,
  FITTING_REPORT_FORMAT,
  IMPACT_COUPLING_REPORT_FORMAT,
  BODY_CHAIN_FORMAT,
} from "./clubFitting";

import fittingDocGolden from "./__fixtures__/fitting_document_golden_v1.json";
import fittingReportGolden from "./__fixtures__/fitting_report_golden_v1.json";
import impactCouplingReportGolden from "./__fixtures__/impact_coupling_report_golden_v1.json";
import bodyChainGolden from "./__fixtures__/body_chain_golden_v1.json";

describe("clubFitting model layer", () => {
  it("parses the canonical fitting document golden fixture", () => {
    const doc = parseFittingDocument(JSON.stringify(fittingDocGolden));
    expect(doc.format).toBe(FITTING_DOCUMENT_FORMAT);
    expect(doc.document_id).toBe("driver-fit-doc-1");
    expect(doc.face.loft_deg).toBe(10.5);
    expect(doc.face.lie_deg).toBe(58.0);
    expect(doc.tip_mass.mass_kg).toBe(0.200);
    expect(doc.provenance.tool_name).toBe("club-tester-generator");
  });

  it("parses the canonical fitting report golden fixture", () => {
    const report = parseFittingReport(JSON.stringify(fittingReportGolden));
    expect(report.format).toBe(FITTING_REPORT_FORMAT);
    expect(report.document_id).toBe("driver-fit-doc-1");
    expect(report.grip.omega_rad_s).toBe(39.0);
    expect(report.baseline.label).toBe("baseline");
    expect(report.baseline.ball_speed_mps).toBeGreaterThan(50);
    expect(report.counterfactuals.length).toBe(3);
    expect(report.counterfactuals[0].label).toBe("loft-plus-2");
    expect(report.counterfactuals[0].deltas_vs_baseline).toBeDefined();
  });

  it("parses the canonical impact coupling report golden fixture", () => {
    const report = parseImpactCouplingReport(JSON.stringify(impactCouplingReportGolden));
    expect(report.format).toBe(IMPACT_COUPLING_REPORT_FORMAT);
    expect(report.baseline.ball_speed_mps).toBeGreaterThan(0);
    expect(report.baseline.decoupling_fraction).toBeGreaterThan(0.95);
    expect(report.counterfactuals.length).toBe(9);
  });

  it("parses the canonical body chain golden fixture and reduces grip boundary", () => {
    const chain = parseBodyChain(JSON.stringify(bodyChainGolden));
    expect(chain.format).toBe(BODY_CHAIN_FORMAT);
    expect(chain.source_id).toBe("mjcf:golfer_upper_body");
    expect(chain.bodies.length).toBe(3);

    const boundary = gripBoundaryReduction(chain, {
      handBodies: ["lead_hand"],
      boundaryJointOf: "lead_hand",
    });
    expect(boundary.effective_mass_kg).toBe(0.8);
    expect(boundary.stiffness_n_m).toBe(80.0);
    expect(boundary.damping_n_s_m).toBe(2.0);
    expect(boundary.provenance).toContain("lead_hand");
  });

  it("supports explicit stiffness/damping overrides in grip boundary reduction", () => {
    const chain = parseBodyChain(JSON.stringify(bodyChainGolden));
    const boundary = gripBoundaryReduction(chain, {
      handBodies: ["lead_arm", "lead_hand"],
      boundaryJointOf: "lead_hand",
      stiffnessOverride: 50000.0,
      dampingOverride: 25.0,
    });
    expect(boundary.effective_mass_kg).toBe(4.0);
    expect(boundary.stiffness_n_m).toBe(50000.0);
    expect(boundary.damping_n_s_m).toBe(25.0);
    expect(boundary.provenance).toContain("overridden");
  });

  it("fails closed on invalid or unknown format", () => {
    expect(() => parseFittingDocument('{"format": "invalid"}')).toThrow();
    expect(() => parseFittingReport('{"format": "invalid"}')).toThrow();
    expect(() => parseImpactCouplingReport('{"format": "invalid"}')).toThrow();
    expect(() => parseBodyChain('{"format": "invalid"}')).toThrow();
  });
});
