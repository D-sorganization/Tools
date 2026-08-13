/**
 * Sectioned derivation parity tests (#4120 V4): section keys mirror
 * the Python side, sections toggle with the configuration, every
 * formula is valid KaTeX, and the plane-gravity mirror matches the
 * pinned Python value.
 */

import katex from "katex";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_DERIVATION_CONFIG,
  derivationSections,
  inPlaneGravity,
} from "./derivationModels";
import { DEFAULT_SCENARIO } from "./impact";

const keysFor = (config = DEFAULT_DERIVATION_CONFIG) =>
  derivationSections(DEFAULT_SCENARIO, config).map((s) => s.key);

describe("section keys (Python parity pins)", () => {
  it("default configuration renders closure/impact/flight", () => {
    expect(keysFor()).toEqual(["closure", "impact", "flight"]);
  });

  it("pendulum sources add the swing section", () => {
    for (const swingSource of ["double_pendulum", "triple_pendulum"]) {
      expect(
        keysFor({ ...DEFAULT_DERIVATION_CONFIG, swingSource }),
      ).toEqual(["closure", "impact", "flight", "swing"]);
    }
  });

  it("triple-pendulum step appears only for the triple source", () => {
    const titles = (swingSource: string) => {
      const sections = derivationSections(DEFAULT_SCENARIO, {
        ...DEFAULT_DERIVATION_CONFIG,
        swingSource,
      });
      return sections[sections.length - 1].steps.map((s) => s.title);
    };
    expect(titles("double_pendulum")).not.toContain(
      "Triple-Pendulum Extension",
    );
    expect(titles("triple_pendulum")).toContain("Triple-Pendulum Extension");
  });

  it("gear-effect step toggles with the flag", () => {
    const impactTitles = (gearEffect: boolean) =>
      derivationSections(DEFAULT_SCENARIO, {
        ...DEFAULT_DERIVATION_CONFIG,
        gearEffect,
      })[1].steps.map((s) => s.title);
    const gearTitle = "Gear Effect — Head Recoil Times CG Depth";
    expect(impactTitles(true)).toContain(gearTitle);
    expect(impactTitles(false)).not.toContain(gearTitle);
  });

  it("the active flight model rewrites the coefficient step", () => {
    const lawTitle = (flightModel: string) =>
      derivationSections(DEFAULT_SCENARIO, {
        ...DEFAULT_DERIVATION_CONFIG,
        flightModel,
      })[2].steps[1].title;
    expect(lawTitle("waterloo_penner")).toContain("Waterloo/Penner");
    expect(lawTitle("nathan")).toContain("Nathan");
    expect(lawTitle("macdonald_hanzely")).toContain("MacDonald-Hanzely");
  });
});

describe("formula validity", () => {
  it("every step renders as strict KaTeX for every configuration", () => {
    const configs = [
      DEFAULT_DERIVATION_CONFIG,
      { ...DEFAULT_DERIVATION_CONFIG, gearEffect: false },
      {
        ...DEFAULT_DERIVATION_CONFIG,
        swingSource: "triple_pendulum" as const,
      },
      { ...DEFAULT_DERIVATION_CONFIG, flightModel: "charry_l3" },
    ];
    for (const config of configs) {
      for (const section of derivationSections(DEFAULT_SCENARIO, config)) {
        for (const step of section.steps) {
          for (const tex of [step.latex, step.values]) {
            expect(() =>
              katex.renderToString(tex, { throwOnError: true }),
            ).not.toThrow();
          }
        }
      }
    }
  });
});

describe("plane gravity mirror", () => {
  it("matches the pinned Python reference value", () => {
    // swing_sim.reference.in_plane_gravity_from_tilts(0, -45°, 0, 9.81)
    const [gx, gy] = inPlaneGravity(0, -45, 0);
    expect(gx).toBeCloseTo(0.0, 6);
    expect(gy).toBeCloseTo(-9.81 * Math.SQRT1_2, 4);
  });

  it("tilted planes feel less in-plane gravity than flat", () => {
    const [, flat] = inPlaneGravity(0, 0, 0);
    const [, tilted] = inPlaneGravity(0, -60, 0);
    expect(Math.abs(tilted)).toBeLessThan(Math.abs(flat));
  });
});
