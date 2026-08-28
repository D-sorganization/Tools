/**
 * Putter-head import — TypeScript mirror of
 * `shared/python/golf_club/putter_head.py` (epic #4800, P3): the
 * `golf_club.putter_head/1` document model (mass, CG, inertia tensor,
 * loft/COR, provenance; JSON wire in `putterHeadWire.ts`, the P2
 * `puttingGreenWire.ts` split), mesh construction through the C1 twin
 * (`volumetrics.meshInertia` — never a second mesh pipeline, same
 * exactly-one density/target-mass selector), and the club-library
 * fallback resolving the `PutterSpec` reconciliation: v2 builds *on*
 * the v1 spec (`putterSpec` recovers it), and a library head carries
 * no tensor, so it strikes bit-identically to P1's default path.
 *
 * Head frame (matching `swing_sim.impact`): x = target line, y = up,
 * z = toe. Quasi-static twist (heavy-hit lumped posture, one-way
 * diagnostic): normal impulse `J = (1 + e) mu v cos(beta)` at offset
 * `r` gives `theta = J r tau_c / (2 I)` per axis — toe about I_yy
 * (+ opens the face), high about I_zz (+ adds loft), `tau_c` the
 * documented ~0.5 ms contact window. Speed loss feeds P1's strike
 * hook via `1/I_eff = (r_t^2/I_yy + r_h^2/I_zz)/r^2`. Python adds an
 * STL file convenience (`putter_head_from_stl`); the web runtime gets
 * triangles + digest from its upload machinery (documented
 * divergence — the math is twinned op-for-op).
 */

import type { Vec3 } from "./club";
import type { Triangle } from "./mesh";
import type { PuttLaunch } from "./puttingGreen";
import {
  DEFAULT_PUTTER_COR,
  DEFAULT_PUTTER_MOI_KG_M2,
  GOLF_BALL_MASS_KG,
  strike,
  type PutterSpec,
} from "./putting";
import { meshInertia } from "./volumetrics";

export const PUTTER_HEAD_FORMAT = "golf_club.putter_head/1";

/** Documented putter-ball contact window [s] (~0.5 ms; see header). */
export const PUTTER_CONTACT_TIME_S = 5.0e-4;

const MM_TO_M = 1e-3;
const SHA256_HEX = /^[0-9a-f]{64}$/;

export type Matrix3 = readonly [Vec3, Vec3, Vec3];

export interface PutterHeadProvenance {
  readonly source_kind: "mesh" | "library";
  readonly mesh_sha256?: string;
  readonly density_kg_m3?: number;
  readonly target_mass_kg?: number;
  readonly library_name?: string;
}

/** PutterSpec v2: the v1 spec fields plus CG, tensor, provenance. */
export interface PutterHeadDocument {
  readonly name: string;
  readonly head_mass_kg: number;
  readonly loft_deg: number;
  readonly cg_m?: Vec3;
  readonly cor: number;
  readonly inertia_at_cg_kg_m2?: Matrix3;
  readonly provenance: PutterHeadProvenance;
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function validateProvenance(provenance: PutterHeadProvenance): void {
  if (provenance.source_kind === "mesh") {
    if (provenance.library_name !== undefined) {
      throw new Error("mesh provenance must not carry library_name");
    }
    if (
      typeof provenance.mesh_sha256 !== "string" ||
      !SHA256_HEX.test(provenance.mesh_sha256)
    ) {
      throw new Error("mesh_sha256 must be 64 lowercase hex characters");
    }
    const hasDensity = provenance.density_kg_m3 !== undefined;
    const hasMass = provenance.target_mass_kg !== undefined;
    if (hasDensity === hasMass) {
      throw new Error(
        "exactly one of density_kg_m3 or target_mass_kg must be given",
      );
    }
    const scale = hasDensity
      ? provenance.density_kg_m3
      : provenance.target_mass_kg;
    if (!(finiteNumber(scale, "inertia scale") > 0)) {
      throw new Error("inertia scale must be > 0");
    }
  } else if (provenance.source_kind === "library") {
    const banned = ["mesh_sha256", "density_kg_m3", "target_mass_kg"] as const;
    for (const name of banned) {
      if (provenance[name] !== undefined) {
        throw new Error(`library provenance must not carry ${name}`);
      }
    }
    const libraryName = provenance.library_name;
    if (
      typeof libraryName !== "string" ||
      libraryName.trim() !== libraryName ||
      !libraryName
    ) {
      throw new Error("library_name must be nonempty and trimmed");
    }
  } else {
    throw new Error("source_kind must be mesh or library");
  }
}

/** Fail-closed document validation (the Python `__post_init__` twin). */
export function validatePutterHead(
  document: PutterHeadDocument,
): PutterHeadDocument {
  // The v1 spec bounds, enforced identically through v2.
  if (!document.name || document.name.trim() !== document.name) {
    throw new Error("putter name must be nonempty and trimmed");
  }
  const mass = finiteNumber(document.head_mass_kg, "head_mass_kg");
  if (mass < 0.1 || mass > 1.0) {
    throw new Error("head mass must be plausible [kg]");
  }
  const loft = finiteNumber(document.loft_deg, "loft_deg");
  if (loft < -2.0 || loft > 10.0) {
    throw new Error("putter loft must be in [-2, 10] deg");
  }
  const cor = finiteNumber(document.cor, "cor");
  if (!(cor > 0 && cor < 1)) {
    throw new Error("COR must be in (0, 1)");
  }
  validateProvenance(document.provenance);
  const hasCg = document.cg_m !== undefined;
  const hasTensor = document.inertia_at_cg_kg_m2 !== undefined;
  if (document.provenance.source_kind === "mesh") {
    if (!hasCg || !hasTensor) {
      throw new Error(
        "mesh-sourced heads must carry cg_m and inertia_at_cg_kg_m2",
      );
    }
    const cg = document.cg_m as Vec3;
    if (cg.length !== 3) throw new Error("cg_m must have three components");
    cg.forEach((value, index) => finiteNumber(value, `cg_m[${index}]`));
    const tensor = document.inertia_at_cg_kg_m2 as Matrix3;
    if (tensor.length !== 3 || tensor.some((row) => row.length !== 3)) {
      throw new Error("inertia_at_cg_kg_m2 must be 3x3");
    }
    for (let j = 0; j < 3; j += 1) {
      for (let k = 0; k < 3; k += 1) {
        const value = finiteNumber(
          tensor[j][k],
          `inertia_at_cg_kg_m2[${j}][${k}]`,
        );
        if (Math.abs(value - tensor[k][j]) > 1e-12) {
          throw new Error("inertia_at_cg_kg_m2 must be symmetric");
        }
      }
    }
    for (const axis of [1, 2]) {
      if (!(tensor[axis][axis] > 0)) {
        throw new Error("twist moments I_yy and I_zz must be > 0");
      }
    }
  } else if (hasCg || hasTensor) {
    throw new Error(
      "library-sourced heads must not carry cg_m or inertia_at_cg_kg_m2",
    );
  }
  return document;
}

/** The v1 `PutterSpec` a v2 document builds on (reconciliation). */
export function putterSpec(document: PutterHeadDocument): PutterSpec {
  return {
    name: document.name,
    headMassKg: document.head_mass_kg,
    loftDeg: document.loft_deg,
    cor: document.cor,
  };
}

/** Build a v2 document from a watertight mesh via the C1 twin. */
export function putterHeadFromMesh(
  name: string,
  triangles: Triangle[],
  opts: {
    meshSha256: string;
    loftDeg: number;
    cor?: number;
    densityKgM3?: number;
    targetMassKg?: number;
  },
): PutterHeadDocument {
  const report = meshInertia(triangles, {
    densityKgM3: opts.densityKgM3,
    massKg: opts.targetMassKg,
  });
  const provenance: PutterHeadProvenance = {
    source_kind: "mesh",
    mesh_sha256: opts.meshSha256,
    ...(opts.densityKgM3 !== undefined
      ? { density_kg_m3: opts.densityKgM3 }
      : { target_mass_kg: opts.targetMassKg }),
  };
  return validatePutterHead({
    name,
    head_mass_kg: report.massKg,
    loft_deg: opts.loftDeg,
    cor: opts.cor ?? DEFAULT_PUTTER_COR,
    provenance,
    cg_m: report.centroidM,
    inertia_at_cg_kg_m2: report.inertiaAtCogKgM2,
  });
}

/** Build the no-mesh fallback document from a club-library putter. */
export function putterHeadFromLibrary(
  libraryName: string,
  opts: { headMassKg: number; loftDeg: number; cor?: number },
): PutterHeadDocument {
  return validatePutterHead({
    name: libraryName,
    head_mass_kg: opts.headMassKg,
    loft_deg: opts.loftDeg,
    cor: opts.cor ?? DEFAULT_PUTTER_COR,
    provenance: { source_kind: "library", library_name: libraryName },
  });
}

/**
 * The scalar MOI for P1's strike hook: `undefined` for a library
 * fallback (strike applies its catalogue default), otherwise the
 * exact directional scalar `r^2 / (r_t^2/I_yy + r_h^2/I_zz)`
 * (`I_yy` at zero offset, where strike ignores the value).
 */
export function headMoiForStrike(
  document: PutterHeadDocument,
  strikeOffsetToeMm = 0.0,
  strikeOffsetHighMm = 0.0,
): number | undefined {
  const tensor = document.inertia_at_cg_kg_m2;
  if (tensor === undefined) return undefined;
  const rT = finiteNumber(strikeOffsetToeMm, "strikeOffsetToeMm") * MM_TO_M;
  const rH = finiteNumber(strikeOffsetHighMm, "strikeOffsetHighMm") * MM_TO_M;
  const moiYY = tensor[1][1];
  const moiZZ = tensor[2][2];
  const rSq = rT * rT + rH * rH;
  if (rSq === 0.0) return moiYY;
  return rSq / ((rT * rT) / moiYY + (rH * rH) / moiZZ);
}

/**
 * Quasi-static face rotation during contact: `faceTwistOpenDeg` about
 * the vertical axis (+ = face opens, toe strike), `loftTwistAddDeg`
 * about the heel-toe axis (+ = loft added, high strike),
 * `normalImpulseNS` the contact impulse J, `headMoiKgM2` the scalar
 * fed to P1's hook (undefined = catalogue default).
 */
export interface PutterTwist {
  faceTwistOpenDeg: number;
  loftTwistAddDeg: number;
  normalImpulseNS: number;
  headMoiKgM2?: number;
}

export interface TwistOptions {
  shaftLeanDeg?: number;
  attackAngleDeg?: number;
  strikeOffsetToeMm?: number;
  strikeOffsetHighMm?: number;
}

/** Quasi-static twist: `theta = J r tau_c / (2 I)` per axis. */
export function twistResponse(
  document: PutterHeadDocument,
  clubheadSpeedMps: number,
  options: TwistOptions = {},
): PutterTwist {
  if (!(clubheadSpeedMps > 0 && clubheadSpeedMps <= 10)) {
    throw new Error("clubheadSpeedMps must be in (0, 10]");
  }
  const shaftLeanDeg = options.shaftLeanDeg ?? 0.0;
  const attackAngleDeg = options.attackAngleDeg ?? 0.0;
  const toeMm = options.strikeOffsetToeMm ?? 0.0;
  const highMm = options.strikeOffsetHighMm ?? 0.0;
  const bounds: Array<[string, number, number]> = [
    ["shaftLeanDeg", shaftLeanDeg, 10.0],
    ["attackAngleDeg", attackAngleDeg, 10.0],
    ["strikeOffsetToeMm", toeMm, 40.0],
    ["strikeOffsetHighMm", highMm, 20.0],
  ];
  for (const [name, value, bound] of bounds) {
    if (!Number.isFinite(value) || Math.abs(value) > bound) {
      throw new Error(`${name} must be within +/-${bound}`);
    }
  }
  const effectiveLoftDeg = document.loft_deg + shaftLeanDeg;
  if (effectiveLoftDeg < -2 || effectiveLoftDeg > 15) {
    throw new Error("effective loft must stay in [-2, 15] deg");
  }
  const tensor = document.inertia_at_cg_kg_m2;
  const moiYY = tensor === undefined ? DEFAULT_PUTTER_MOI_KG_M2 : tensor[1][1];
  const moiZZ = tensor === undefined ? DEFAULT_PUTTER_MOI_KG_M2 : tensor[2][2];
  const moiHook = headMoiForStrike(document, toeMm, highMm);

  const rT = toeMm * MM_TO_M;
  const rH = highMm * MM_TO_M;
  const offsetR = Math.hypot(rT, rH);
  const mass = document.head_mass_kg;
  const scalarMoi = moiHook ?? DEFAULT_PUTTER_MOI_KG_M2;
  const massEff =
    offsetR === 0.0
      ? mass
      : 1.0 / (1.0 / mass + (offsetR * offsetR) / scalarMoi);
  const beta = ((effectiveLoftDeg - attackAngleDeg) * Math.PI) / 180.0;
  const reducedMass =
    (massEff * GOLF_BALL_MASS_KG) / (massEff + GOLF_BALL_MASS_KG);
  const impulse =
    (1.0 + document.cor) * reducedMass * clubheadSpeedMps * Math.cos(beta);

  const halfWindow = PUTTER_CONTACT_TIME_S / 2.0;
  const toDeg = 180.0 / Math.PI;
  return {
    faceTwistOpenDeg: ((impulse * rT) / moiYY) * halfWindow * toDeg,
    loftTwistAddDeg: ((impulse * rH) / moiZZ) * halfWindow * toDeg,
    normalImpulseNS: impulse,
    ...(moiHook !== undefined ? { headMoiKgM2: moiHook } : {}),
  };
}

export type StrikeWithHeadOptions = TwistOptions & {
  aimDeg?: number;
  faceAngleDeg?: number;
  pathAngleDeg?: number;
};

/** The P1 launch plus this module's twist diagnostic. */
export interface PutterStrikeResult {
  launch: PuttLaunch;
  twist: PutterTwist;
}

/**
 * Solve the P1 impact with this head's MOI feeding the explicit hook;
 * a library-fallback head passes undefined and reproduces P1's
 * catalogue-default results field-for-field (a test gate).
 */
export function strikeWithHead(
  document: PutterHeadDocument,
  clubheadSpeedMps: number,
  shaftLeanDeg = 0.0,
  options: StrikeWithHeadOptions = {},
): PutterStrikeResult {
  const toeMm = options.strikeOffsetToeMm ?? 0.0;
  const highMm = options.strikeOffsetHighMm ?? 0.0;
  const launch = strike(putterSpec(document), clubheadSpeedMps, shaftLeanDeg, {
    aimDeg: options.aimDeg,
    faceAngleDeg: options.faceAngleDeg,
    pathAngleDeg: options.pathAngleDeg,
    attackAngleDeg: options.attackAngleDeg,
    strikeOffsetToeMm: toeMm,
    strikeOffsetHighMm: highMm,
    headMoiKgM2: headMoiForStrike(document, toeMm, highMm),
  });
  const twist = twistResponse(document, clubheadSpeedMps, {
    shaftLeanDeg,
    attackAngleDeg: options.attackAngleDeg,
    strikeOffsetToeMm: toeMm,
    strikeOffsetHighMm: highMm,
  });
  return { launch, twist };
}
