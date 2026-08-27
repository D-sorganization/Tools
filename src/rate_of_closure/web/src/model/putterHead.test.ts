/**
 * Putter-head import gates (epic #4800, P3) — mirrors
 * `tests/shared/python/golf_club/test_putter_head.py` test-for-test.
 * Analytic gates first: box inertia closed forms, the
 * `J r tau / (2 I)` twist form, and the P1 default-MOI fallback.
 */

import { describe, expect, it } from "vitest";

import type { Vec3 } from "./club";
import type { Triangle } from "./mesh";
import {
  GOLF_BALL_MASS_KG,
  DEFAULT_PUTTER_MOI_KG_M2,
  MINIMAL_PUTTERS,
  strike,
} from "./putting";
import {
  PUTTER_CONTACT_TIME_S,
  PUTTER_HEAD_FORMAT,
  headMoiForStrike,
  putterHeadFromLibrary,
  putterHeadFromMesh,
  putterSpec,
  strikeWithHead,
  twistResponse,
  validatePutterHead,
  type PutterHeadDocument,
} from "./putterHead";
import { putterHeadFromJson, putterHeadToJson } from "./putterHeadWire";
import { meshInertia } from "./volumetrics";

const SHA = "a".repeat(64);

/** Float32-exact blade-like box (31.25 x 31.25 x 125 mm). */
const BLADE_EXTENTS: Vec3 = [0.03125, 0.03125, 0.125];
const BLADE_MASS_KG = 0.35;

function boxMesh(extents: Vec3, center: Vec3): Triangle[] {
  const [hx, hy, hz] = extents.map((e) => e / 2);
  const corners: Vec3[] = [];
  for (const sx of [-hx, hx]) {
    for (const sy of [-hy, hy]) {
      for (const sz of [-hz, hz]) {
        corners.push([center[0] + sx, center[1] + sy, center[2] + sz]);
      }
    }
  }
  const faces: Array<[number, number, number, number]> = [
    [0, 1, 3, 2],
    [4, 6, 7, 5],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 5, 7, 3],
  ];
  const triangles: Triangle[] = [];
  for (const [a, b, c, d] of faces) {
    triangles.push([corners[a], corners[b], corners[c]]);
    triangles.push([corners[a], corners[c], corners[d]]);
  }
  return triangles;
}

function boxInertiaDiag(extents: Vec3, mass: number): Vec3 {
  const [ax, ay, az] = extents;
  return [
    (mass / 12) * (ay * ay + az * az),
    (mass / 12) * (ax * ax + az * az),
    (mass / 12) * (ax * ax + ay * ay),
  ];
}

function bladeDocument(): PutterHeadDocument {
  return putterHeadFromMesh("Milled Blade", boxMesh(BLADE_EXTENTS, [0, 0, 0]), {
    meshSha256: SHA,
    loftDeg: 3.0,
    targetMassKg: BLADE_MASS_KG,
  });
}

const BLADE_V1 = MINIMAL_PUTTERS.find((p) => p.name === "Blade Putter")!;

function libraryDocument(): PutterHeadDocument {
  return putterHeadFromLibrary(BLADE_V1.name, {
    headMassKg: BLADE_V1.headMassKg,
    loftDeg: BLADE_V1.loftDeg,
  });
}

describe("meshInertia (C1 twin)", () => {
  it("recovers the analytic cube tensor m L^2 / 6 exactly", () => {
    const report = meshInertia(boxMesh([0.4, 0.4, 0.4], [0, 0, 0]), {
      densityKgM3: 1234.5,
    });
    const expected = (report.massKg * 0.4 * 0.4) / 6;
    for (let j = 0; j < 3; j += 1) {
      for (let k = 0; k < 3; k += 1) {
        expect(report.inertiaAtCogKgM2[j][k]).toBeCloseTo(
          j === k ? expected : 0,
          12,
        );
      }
    }
  });

  it("recovers the offset-box closed form and centroid", () => {
    const extents: Vec3 = [0.3, 0.1, 0.05];
    const center: Vec3 = [1.7, -2.2, 0.9];
    const report = meshInertia(boxMesh(extents, center), { massKg: 2.5 });
    const expected = boxInertiaDiag(extents, 2.5);
    report.centroidM.forEach((c, i) => expect(c).toBeCloseTo(center[i], 10));
    for (let axis = 0; axis < 3; axis += 1) {
      expect(report.inertiaAtCogKgM2[axis][axis]).toBeCloseTo(
        expected[axis],
        12,
      );
    }
  });

  it("enforces the exactly-one scale selector", () => {
    const mesh = boxMesh(BLADE_EXTENTS, [0, 0, 0]);
    expect(() => meshInertia(mesh, {})).toThrow(/exactly one/);
    expect(() =>
      meshInertia(mesh, { densityKgM3: 8000, massKg: 0.35 }),
    ).toThrow(/exactly one/);
  });

  it("refuses an open mesh", () => {
    const mesh = boxMesh(BLADE_EXTENTS, [0, 0, 0]).slice(0, -1);
    expect(() => meshInertia(mesh, { massKg: 0.35 })).toThrow(/watertight/);
  });
});

describe("provenance validation", () => {
  it("mesh requires sha and exactly one selector", () => {
    expect(() =>
      validatePutterHead({
        ...bladeDocument(),
        provenance: { source_kind: "mesh", density_kg_m3: 8000 },
      }),
    ).toThrow(/mesh_sha256/);
    expect(() =>
      validatePutterHead({
        ...bladeDocument(),
        provenance: { source_kind: "mesh", mesh_sha256: SHA },
      }),
    ).toThrow(/exactly one/);
    expect(() =>
      validatePutterHead({
        ...bladeDocument(),
        provenance: {
          source_kind: "mesh",
          mesh_sha256: SHA,
          density_kg_m3: 8000,
          target_mass_kg: 0.35,
        },
      }),
    ).toThrow(/exactly one/);
    expect(() =>
      validatePutterHead({
        ...bladeDocument(),
        provenance: {
          source_kind: "mesh",
          mesh_sha256: SHA,
          density_kg_m3: 8000,
          library_name: "Blade Putter",
        },
      }),
    ).toThrow(/library_name/);
  });

  it("library requires the name and nothing else", () => {
    expect(() =>
      validatePutterHead({
        ...libraryDocument(),
        provenance: { source_kind: "library" },
      }),
    ).toThrow(/library_name/);
    expect(() =>
      validatePutterHead({
        ...libraryDocument(),
        provenance: {
          source_kind: "library",
          library_name: "Blade Putter",
          mesh_sha256: SHA,
        },
      }),
    ).toThrow(/must not carry/);
  });

  it("unknown source kinds are refused", () => {
    expect(() =>
      validatePutterHead({
        ...libraryDocument(),
        provenance: {
          source_kind: "guessed" as "library",
          library_name: "x",
        },
      }),
    ).toThrow(/source_kind/);
  });
});

describe("mesh construction", () => {
  it("box head matches the closed-form inertia", () => {
    const document = bladeDocument();
    expect(document.head_mass_kg).toBeCloseTo(BLADE_MASS_KG, 12);
    document.cg_m!.forEach((c) => expect(c).toBeCloseTo(0, 12));
    const expected = boxInertiaDiag(BLADE_EXTENTS, BLADE_MASS_KG);
    for (let axis = 0; axis < 3; axis += 1) {
      expect(document.inertia_at_cg_kg_m2![axis][axis]).toBeCloseTo(
        expected[axis],
        12,
      );
    }
  });

  it("density selector sets mass from the enclosed volume", () => {
    const density = 2800;
    const document = putterHeadFromMesh(
      "Cast Blade",
      boxMesh(BLADE_EXTENTS, [0, 0, 0]),
      { meshSha256: SHA, loftDeg: 3.0, densityKgM3: density },
    );
    const volume = BLADE_EXTENTS[0] * BLADE_EXTENTS[1] * BLADE_EXTENTS[2];
    expect(document.head_mass_kg).toBeCloseTo(density * volume, 12);
    expect(document.provenance.density_kg_m3).toBe(density);
    expect(document.provenance.target_mass_kg).toBeUndefined();
  });

  it("mesh documents require mass properties", () => {
    const document = bladeDocument();
    const rest = {
      name: document.name,
      head_mass_kg: document.head_mass_kg,
      loft_deg: document.loft_deg,
      cor: document.cor,
      provenance: document.provenance,
    };
    expect(() => validatePutterHead(rest)).toThrow(/must carry/);
  });

  it("enforces the v1 spec bounds through v2", () => {
    const mesh = boxMesh(BLADE_EXTENTS, [0, 0, 0]);
    expect(() =>
      putterHeadFromMesh("x", mesh, {
        meshSha256: SHA,
        loftDeg: 45.0,
        targetMassKg: 0.35,
      }),
    ).toThrow(/loft/);
    expect(() =>
      putterHeadFromMesh("x", mesh, {
        meshSha256: SHA,
        loftDeg: 3.0,
        targetMassKg: 5.0,
      }),
    ).toThrow(/mass/);
  });
});

describe("library construction", () => {
  it("carries no tensor and recovers the v1 record", () => {
    const document = libraryDocument();
    expect(document.cg_m).toBeUndefined();
    expect(document.inertia_at_cg_kg_m2).toBeUndefined();
    expect(document.provenance).toEqual({
      source_kind: "library",
      library_name: "Blade Putter",
    });
    expect(putterSpec(document)).toEqual(BLADE_V1);
  });

  it("refuses mass properties on a library document", () => {
    expect(() =>
      validatePutterHead({
        ...libraryDocument(),
        inertia_at_cg_kg_m2: [
          [1e-4, 0, 0],
          [0, 1e-4, 0],
          [0, 0, 1e-4],
        ],
      }),
    ).toThrow(/must not carry/);
  });
});

describe("wire golf_club.putter_head/1", () => {
  it("mesh round-trip is byte-deterministic with sorted keys", () => {
    const document = bladeDocument();
    const text = putterHeadToJson(document);
    const parsed = putterHeadFromJson(text);
    expect(parsed).toEqual(document);
    expect(putterHeadToJson(parsed)).toBe(text);
    const payload = JSON.parse(text) as Record<string, unknown>;
    expect(payload.format).toBe(PUTTER_HEAD_FORMAT);
    expect(Object.keys(payload)).toEqual([...Object.keys(payload)].sort());
  });

  it("library round-trip omits the mesh-only fields", () => {
    const document = libraryDocument();
    const text = putterHeadToJson(document);
    const payload = JSON.parse(text) as Record<string, unknown>;
    expect(payload.cg_m).toBeUndefined();
    expect(payload.inertia_at_cg_kg_m2).toBeUndefined();
    expect(payload.provenance).toEqual({
      source_kind: "library",
      library_name: "Blade Putter",
    });
    expect(putterHeadFromJson(text)).toEqual(document);
  });

  it("refuses unknown fields at both levels", () => {
    const text = putterHeadToJson(libraryDocument());
    const withExtra = JSON.parse(text) as Record<string, unknown>;
    withExtra.smoothing = true;
    expect(() => putterHeadFromJson(JSON.stringify(withExtra))).toThrow(
      /unknown fields/,
    );
    const nested = JSON.parse(text) as {
      provenance: Record<string, unknown>;
    };
    nested.provenance.vendor = "acme";
    expect(() => putterHeadFromJson(JSON.stringify(nested))).toThrow(
      /unknown fields/,
    );
  });

  it("refuses wrong formats and non-finite numbers", () => {
    const payload = JSON.parse(putterHeadToJson(libraryDocument())) as Record<
      string,
      unknown
    >;
    payload.format = "golf_club.putter_head/2";
    expect(() => putterHeadFromJson(JSON.stringify(payload))).toThrow(/format/);
    expect(() =>
      putterHeadToJson({
        ...libraryDocument(),
        head_mass_kg: Number.POSITIVE_INFINITY,
      }),
    ).toThrow(/finite/);
  });
});

describe("headMoiForStrike", () => {
  it("returns undefined for the library fallback", () => {
    expect(headMoiForStrike(libraryDocument(), 10, 5)).toBeUndefined();
  });

  it("single-axis offsets pick the matching moment", () => {
    const document = bladeDocument();
    const [, moiYY, moiZZ] = boxInertiaDiag(BLADE_EXTENTS, BLADE_MASS_KG);
    expect(headMoiForStrike(document)).toBeCloseTo(moiYY, 12);
    expect(headMoiForStrike(document, 10, 0)).toBeCloseTo(moiYY, 12);
    expect(headMoiForStrike(document, 0, 8)).toBeCloseTo(moiZZ, 12);
  });

  it("combined offsets match the directional closed form", () => {
    const document = bladeDocument();
    const [, moiYY, moiZZ] = boxInertiaDiag(BLADE_EXTENTS, BLADE_MASS_KG);
    const rT = 10e-3;
    const rH = 6e-3;
    const expected =
      (rT * rT + rH * rH) / ((rT * rT) / moiYY + (rH * rH) / moiZZ);
    expect(headMoiForStrike(document, 10, 6)).toBeCloseTo(expected, 12);
  });

  it("feeds the P1 hook exactly", () => {
    const document = bladeDocument();
    const result = strikeWithHead(document, 2.0, 0.0, {
      strikeOffsetToeMm: 10,
    });
    const expected = strike(putterSpec(document), 2.0, 0.0, {
      strikeOffsetToeMm: 10,
      headMoiKgM2: headMoiForStrike(document, 10, 0),
    });
    expect(result.launch).toEqual(expected);
  });
});

describe("twist analytic gates", () => {
  it("symmetric head, center strike: zero twist", () => {
    const twist = twistResponse(bladeDocument(), 2.0);
    expect(twist.faceTwistOpenDeg).toBe(0);
    expect(twist.loftTwistAddDeg).toBe(0);
  });

  it("sign flips toe vs heel and high vs low", () => {
    const document = bladeDocument();
    const toe = twistResponse(document, 2.0, { strikeOffsetToeMm: 10 });
    const heel = twistResponse(document, 2.0, { strikeOffsetToeMm: -10 });
    expect(toe.faceTwistOpenDeg).toBeGreaterThan(0);
    expect(heel.faceTwistOpenDeg).toBe(-toe.faceTwistOpenDeg);
    const high = twistResponse(document, 2.0, { strikeOffsetHighMm: 6 });
    const low = twistResponse(document, 2.0, { strikeOffsetHighMm: -6 });
    expect(high.loftTwistAddDeg).toBeGreaterThan(0);
    expect(low.loftTwistAddDeg).toBe(-high.loftTwistAddDeg);
  });

  it("matches the offset*impulse/MOI closed form", () => {
    const document = bladeDocument();
    const [, moiYY] = boxInertiaDiag(BLADE_EXTENTS, BLADE_MASS_KG);
    const speed = 2.0;
    const toeMm = 10.0;
    const twist = twistResponse(document, speed, {
      strikeOffsetToeMm: toeMm,
    });
    const rT = toeMm * 1e-3;
    const massEff = 1 / (1 / BLADE_MASS_KG + (rT * rT) / moiYY);
    const reduced =
      (massEff * GOLF_BALL_MASS_KG) / (massEff + GOLF_BALL_MASS_KG);
    const impulse =
      (1 + 0.78) * reduced * speed * Math.cos((3.0 * Math.PI) / 180);
    const expected =
      (((impulse * rT) / moiYY) * (PUTTER_CONTACT_TIME_S / 2) * 180) / Math.PI;
    expect(twist.normalImpulseNS).toBeCloseTo(impulse, 12);
    expect(twist.faceTwistOpenDeg).toBeCloseTo(expected, 12);
  });

  it("a higher-MOI head twists less", () => {
    const blade = bladeDocument();
    const mallet = putterHeadFromMesh(
      "Deep Mallet",
      boxMesh([0.125, 0.03125, 0.125], [0, 0, 0]),
      { meshSha256: SHA, loftDeg: 3.0, targetMassKg: 0.36 },
    );
    const bladeTwist = twistResponse(blade, 2.0, { strikeOffsetToeMm: 10 });
    const malletTwist = twistResponse(mallet, 2.0, {
      strikeOffsetToeMm: 10,
    });
    expect(malletTwist.faceTwistOpenDeg).toBeGreaterThan(0);
    expect(malletTwist.faceTwistOpenDeg).toBeLessThan(
      bladeTwist.faceTwistOpenDeg,
    );
  });

  it("library fallback uses the catalogue default", () => {
    const document = libraryDocument();
    const speed = 2.0;
    const toeMm = 10.0;
    const twist = twistResponse(document, speed, {
      strikeOffsetToeMm: toeMm,
    });
    const rT = toeMm * 1e-3;
    const massEff =
      1 / (1 / document.head_mass_kg + (rT * rT) / DEFAULT_PUTTER_MOI_KG_M2);
    const reduced =
      (massEff * GOLF_BALL_MASS_KG) / (massEff + GOLF_BALL_MASS_KG);
    const impulse =
      (1 + 0.78) * reduced * speed * Math.cos((3.0 * Math.PI) / 180);
    const expected =
      (((impulse * rT) / DEFAULT_PUTTER_MOI_KG_M2) *
        (PUTTER_CONTACT_TIME_S / 2) *
        180) /
      Math.PI;
    expect(twist.headMoiKgM2).toBeUndefined();
    expect(twist.faceTwistOpenDeg).toBeCloseTo(expected, 12);
  });

  it("twist stays small at putt speeds", () => {
    const twist = twistResponse(bladeDocument(), 3.0, {
      strikeOffsetToeMm: 20,
      strikeOffsetHighMm: 10,
    });
    expect(Math.abs(twist.faceTwistOpenDeg)).toBeLessThan(1);
    expect(Math.abs(twist.loftTwistAddDeg)).toBeLessThan(1);
  });

  it("rejects out-of-range inputs", () => {
    const document = bladeDocument();
    expect(() => twistResponse(document, 0.0)).toThrow();
    expect(() => twistResponse(document, 2.0, { shaftLeanDeg: 20 })).toThrow();
    expect(() =>
      twistResponse(document, 2.0, { strikeOffsetToeMm: 50 }),
    ).toThrow();
    expect(() =>
      twistResponse(document, 2.0, { strikeOffsetHighMm: -30 }),
    ).toThrow();
  });
});

describe("library fallback reconciliation", () => {
  it("library head reproduces P1's default-MOI behavior exactly", () => {
    const document = libraryDocument();
    const cases: Array<[number, number, number]> = [
      [0.5, 0, 0],
      [1.8, 10, 0],
      [2.5, -8, 6],
      [3.2, 15, -5],
    ];
    for (const [speed, toe, high] of cases) {
      const viaHead = strikeWithHead(document, speed, 0.0, {
        strikeOffsetToeMm: toe,
        strikeOffsetHighMm: high,
      }).launch;
      const viaV1Default = strike(BLADE_V1, speed, 0.0, {
        strikeOffsetToeMm: toe,
        strikeOffsetHighMm: high,
      });
      expect(viaHead).toEqual(viaV1Default);
    }
  });

  it("mesh heads depart from the default off-center only", () => {
    const document = bladeDocument();
    const specV1 = putterSpec(document);
    expect(strikeWithHead(document, 2.0).launch).toEqual(strike(specV1, 2.0));
    const offCenter = strikeWithHead(document, 2.0, 0.0, {
      strikeOffsetToeMm: 10,
    }).launch;
    expect(offCenter).not.toEqual(
      strike(specV1, 2.0, 0.0, { strikeOffsetToeMm: 10 }),
    );
  });
});

describe("strikeWithHead", () => {
  it("passes the full stroke through to P1 with the twist attached", () => {
    const document = bladeDocument();
    const result = strikeWithHead(document, 2.0, -1.0, {
      aimDeg: 2.0,
      faceAngleDeg: 1.0,
      pathAngleDeg: -1.0,
      attackAngleDeg: 1.5,
      strikeOffsetToeMm: 8.0,
    });
    const expected = strike(putterSpec(document), 2.0, -1.0, {
      aimDeg: 2.0,
      faceAngleDeg: 1.0,
      pathAngleDeg: -1.0,
      attackAngleDeg: 1.5,
      strikeOffsetToeMm: 8.0,
      headMoiKgM2: headMoiForStrike(document, 8.0, 0.0),
    });
    expect(result.launch).toEqual(expected);
    expect(result.twist.faceTwistOpenDeg).toBeGreaterThan(0);
  });
});
