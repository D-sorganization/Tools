import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RotatingBaseStudy } from "../src/components/RotatingBaseStudy";
import {
  ROTATING_BASE_BOUNDARIES,
  ROTATING_BASE_SOURCE_REVISION,
  ROTATING_BASE_STUDY,
  registeredCase,
  validateRotatingBaseStudy,
} from "../src/rotatingBaseStudy";

describe("qualified rotating-base evidence browser", () => {
  it("retains the exact registered design and adverse rows", () => {
    expect(ROTATING_BASE_SOURCE_REVISION).toHaveLength(40);
    expect(ROTATING_BASE_STUDY.cases).toHaveLength(18);
    expect(ROTATING_BASE_STUDY.valid_case_count).toBe(13);
    expect(ROTATING_BASE_STUDY.cases.filter((item) => !item.valid).map((item) => item.case_index)).toEqual([6, 7, 8, 15, 16]);
    expect(ROTATING_BASE_BOUNDARIES.coaching_recommendation).toBe("unsupported");
  });

  it("selects every registered axis without favorable filtering", () => {
    const adverse = registeredCase("decelerate", "absolute_club_rate", 3.5);

    expect(adverse.case_index).toBe(16);
    expect(adverse.valid).toBe(false);
    expect(adverse.exclusion_reasons).toEqual([
      "registered_peak_grip_force_ceiling_exceeded",
    ]);
  });

  it("fails closed when adverse evidence or a killswitch channel is removed", () => {
    const missingReason = structuredClone(ROTATING_BASE_STUDY);
    missingReason.cases[6].exclusion_reasons = [];
    expect(() => validateRotatingBaseStudy(missingReason)).toThrow(
      "validity and exclusion reasons disagree",
    );

    const missingChannel = structuredClone(ROTATING_BASE_STUDY) as unknown as {
      same_state_killswitch: { channels: Record<string, unknown> };
    };
    delete missingChannel.same_state_killswitch.channels.bilateral_wrist;
    expect(() => validateRotatingBaseStudy(missingChannel)).toThrow(
      "killswitch channels do not match",
    );
  });

  it("renders scientific boundaries, diagnostics, and killswitches", () => {
    const html = renderToString(<RotatingBaseStudy />);

    expect(html).toContain("Qualified Rotating-Base Study");
    expect(html).toContain("nonanatomical");
    expect(html).toContain("no governed human validation");
    expect(html).toContain("no coaching recommendation");
    expect(html).toContain("Exact Same-State Killswitches");
    expect(html).toContain("Contact Work on Club");
    expect(html).toContain("Work–Energy Closure");
  });
});
