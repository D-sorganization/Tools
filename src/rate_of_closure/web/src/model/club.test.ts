/**
 * Club model parity tests — the numeric pins mirror
 * `tests/rate_of_closure/test_club.py::TestParity` (and friends)
 * verbatim, keeping the TypeScript twin in lock-step with Python.
 */

import { describe, expect, it } from "vitest";

import {
  buildParametricHead,
  CLUB_LIBRARY,
  clubInertia,
  faceNormalAtOffset,
  faceSagitta,
  getClub,
  parametricHeadMesh,
  REFERENCE_HEAD_MASS_KG,
} from "./club";

const DRIVER = "Driver 10.5°";

describe("club library", () => {
  it("holds exactly sixteen clubs in ladder order", () => {
    expect(CLUB_LIBRARY).toHaveLength(16);
    expect(CLUB_LIBRARY[0].name).toBe("Driver 9.5°");
    expect(CLUB_LIBRARY[14].name).toBe("Blade Putter");
    expect(CLUB_LIBRARY[15].name).toBe("Mallet Putter");
  });

  it("normalizes the driver source row to SI", () => {
    const driver = getClub(DRIVER);
    expect(driver.lengthM).toBeCloseTo(45.5 * 0.0254, 12);
    expect(driver.headMassKg).toBeCloseTo(0.2, 12);
    expect(driver.moiAboutShaftKgM2).toBeCloseTo(5.2e-4, 12);
    expect(driver.cgDepthM).toBeCloseTo(0.025, 12);
    expect(driver.lieDeg).toBe(56);
  });

  it("gives woods curved faces and irons flat ones", () => {
    for (const club of CLUB_LIBRARY) {
      const curved = ["Driver", "Wood", "Hybrid"].includes(club.clubType);
      expect(club.faceBulgeRadiusM !== null).toBe(curved);
      expect(club.faceRollRadiusM !== null).toBe(curved);
    }
  });

  it("rejects unknown clubs", () => {
    expect(() => getClub("2-Iron")).toThrow(/unknown club/);
  });
});

describe("club inertia", () => {
  it("matches the hand-computed composition (1 m, 200 g head)", () => {
    // total = 0.35; balance = 0.25625/0.35; I_grip = 0.234375;
    // I_shaft = 5e-4 + 0.1·0.006² + 0.05·0.011² = 5.0965e-4.
    const hand = {
      ...getClub(DRIVER),
      lengthM: 1.0,
      headMassKg: 0.2,
      moiAboutShaftKgM2: 5.0e-4,
    };
    const inertia = clubInertia(hand, 0.1, 0.05);
    expect(inertia.totalMassKg).toBeCloseTo(0.35, 12);
    expect(inertia.balancePointM).toBeCloseTo(0.25625 / 0.35, 12);
    expect(inertia.moiAboutGripKgM2).toBeCloseTo(0.234375, 12);
    expect(inertia.moiAboutShaftKgM2).toBeCloseTo(5.0965e-4, 12);
  });

  it("pins the driver inertia against pytest", () => {
    const inertia = clubInertia(getClub(DRIVER));
    expect(inertia.totalMassKg).toBeCloseTo(0.325, 12);
    expect(inertia.balancePointM).toBeCloseTo(0.863780769230769, 12);
    expect(inertia.moiAboutGripKgM2).toBeCloseTo(0.301561226916667, 12);
    expect(inertia.moiAboutShaftKgM2).toBeCloseTo(5.2875e-4, 12);
  });

  it("validates component masses", () => {
    expect(() => clubInertia(getClub(DRIVER), 0)).toThrow(/out of range/);
    expect(() => clubInertia(getClub(DRIVER), 0.075, 0.5)).toThrow(
      /out of range/,
    );
  });
});

describe("face curvature", () => {
  it("pins the driver sagitta against pytest", () => {
    const driver = getClub(DRIVER);
    expect(faceSagitta(driver, 0.02, 0)).toBeCloseTo(6.6740905808465589e-4, 15);
    expect(faceSagitta(driver, 0.02, 0.01)).toBeCloseTo(
      8.4603746542022407e-4,
      15,
    );
  });

  it("is zero for flat faces", () => {
    expect(faceSagitta(getClub("7-Iron"), 0.02, 0.01)).toBe(0);
  });

  it("pins the driver face normal at (20, 10) mm against pytest", () => {
    const normal = faceNormalAtOffset(getClub(DRIVER), 20, 10);
    expect(normal[0]).toBeCloseTo(0.973950411287592, 12);
    expect(normal[1]).toBeCloseTo(0.216752844685502, 12);
    expect(normal[2]).toBeCloseTo(0.066624324938218, 12);
  });

  it("returns pure loft at center and for flat faces", () => {
    for (const name of [DRIVER, "7-Iron"]) {
      const club = getClub(name);
      const lam = (club.loftDeg * Math.PI) / 180;
      const normal = faceNormalAtOffset(club, 0, 0);
      expect(normal[0]).toBeCloseTo(Math.cos(lam), 12);
      expect(normal[1]).toBeCloseTo(Math.sin(lam), 12);
      expect(normal[2]).toBeCloseTo(0, 12);
    }
    const iron = faceNormalAtOffset(getClub("7-Iron"), 15, 8);
    const center = faceNormalAtOffset(getClub("7-Iron"), 0, 0);
    expect(iron).toEqual(center);
  });

  it("opens the face toward the toe under bulge", () => {
    const toe = faceNormalAtOffset(getClub(DRIVER), 20, 0);
    const heel = faceNormalAtOffset(getClub(DRIVER), -20, 0);
    expect(toe[2]).toBeGreaterThan(0);
    expect(heel[2]).toBeLessThan(0);
    expect(toe[2]).toBeCloseTo(-heel[2], 12);
  });

  it("rejects offsets outside the curvature radius", () => {
    expect(() => faceNormalAtOffset(getClub(DRIVER), 400, 0)).toThrow(
      /bulge radius/,
    );
  });
});

describe("parametric head", () => {
  it("is closed and deterministic", () => {
    const first = buildParametricHead(getClub(DRIVER));
    const second = buildParametricHead(getClub(DRIVER));
    expect(first).toHaveLength(288);
    expect(first).toEqual(second);
  });

  it("pins the driver mesh forward extent and a face vertex", () => {
    const flat = buildParametricHead(getClub(DRIVER)).flat();
    const xMax = Math.max(...flat.map((v) => v[0]));
    expect(xMax).toBeCloseTo(0.058722579135751, 12);
    const target = [0.049434717761548, -0.001031464094849, 0.058];
    const hit = flat.some(
      (v) =>
        Math.abs(v[0] - target[0]) +
          Math.abs(v[1] - target[1]) +
          Math.abs(v[2] - target[2]) <
        1e-12,
    );
    expect(hit).toBe(true);
  });

  it("scales the envelope with head mass", () => {
    const wood = getClub("3-Wood");
    const scale = (wood.headMassKg / REFERENCE_HEAD_MASS_KG) ** (1 / 3);
    const flat = buildParametricHead(wood).flat();
    const zs = flat.map((v) => v[2]);
    expect(Math.max(...zs) - Math.min(...zs)).toBeCloseTo(0.124 * scale, 12);
  });

  it("produces unit normals in the renderable mesh", () => {
    const mesh = parametricHeadMesh(getClub("3-Wood"));
    for (const n of mesh.normals) {
      expect(Math.hypot(...n)).toBeCloseTo(1, 9);
    }
  });
});
