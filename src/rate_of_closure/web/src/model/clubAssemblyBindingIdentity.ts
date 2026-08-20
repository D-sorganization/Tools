/** Cross-language canonical identity bytes for selected specs and assemblies. */

import { type ClubSpec } from "./club";

export interface ClubAssemblyBindingDigestRuntime {
  sha256Hex: (payload: ArrayBuffer) => Promise<string>;
}

const SHA256_HEX = /^[a-f0-9]{64}$/u;

async function browserSha256Hex(payload: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("browser SHA-256 capability is unavailable");
  const digest = await subtle.digest("SHA-256", payload);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}

export function browserBindingDigestRuntime(): ClubAssemblyBindingDigestRuntime {
  return { sha256Hex: browserSha256Hex };
}

function float64BigEndianHex(value: number): string {
  if (!Number.isFinite(value)) {
    throw new Error("identity input numbers must be finite");
  }
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function normalizeIdentityValue(value: unknown): unknown {
  if (
    value === null ||
    typeof value === "boolean" ||
    typeof value === "string"
  ) {
    return value;
  }
  if (typeof value === "number") {
    return { $float64_be: float64BigEndianHex(value) };
  }
  if (Array.isArray(value)) return value.map(normalizeIdentityValue);
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, normalizeIdentityValue(child)]),
    );
  }
  throw new Error(`unsupported identity input type ${typeof value}`);
}

/** Return canonical UTF-8 bytes shared with the Python identity contract. */
export function canonicalIdentityPayload(value: unknown): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify(normalizeIdentityValue(value)),
  );
}

/** Return the complete snake-case ClubSpec snapshot used by the binding wire. */
export function selectedSpecSnapshot(spec: ClubSpec): Record<string, unknown> {
  return {
    name: spec.name,
    club_type: spec.clubType,
    length_m: spec.lengthM,
    head_mass_kg: spec.headMassKg,
    loft_deg: spec.loftDeg,
    lie_deg: spec.lieDeg,
    moi_about_shaft_kg_m2: spec.moiAboutShaftKgM2,
    cg_depth_m: spec.cgDepthM,
    cg_height_m: spec.cgHeightM,
    face_bulge_radius_m: spec.faceBulgeRadiusM,
    face_roll_radius_m: spec.faceRollRadiusM,
    head_style: spec.headStyle ?? "Auto",
  };
}

export function clubSpecIdentityPayload(spec: ClubSpec): Uint8Array {
  return canonicalIdentityPayload(selectedSpecSnapshot(spec));
}

export function clubAssemblyIdentityPayload(assembly: unknown): Uint8Array {
  return canonicalIdentityPayload(assembly);
}

async function digest(
  payload: Uint8Array,
  runtime: ClubAssemblyBindingDigestRuntime,
): Promise<string> {
  const digestBytes = new Uint8Array(payload.byteLength);
  digestBytes.set(payload);
  const sha256 = await runtime.sha256Hex(digestBytes.buffer);
  if (!SHA256_HEX.test(sha256)) {
    throw new Error("identity digest must be lowercase SHA-256");
  }
  return sha256;
}

export async function clubSpecIdentity(
  spec: ClubSpec,
  runtime: ClubAssemblyBindingDigestRuntime = browserBindingDigestRuntime(),
): Promise<string> {
  return digest(clubSpecIdentityPayload(spec), runtime);
}

export async function clubAssemblyIdentity(
  assembly: unknown,
  runtime: ClubAssemblyBindingDigestRuntime = browserBindingDigestRuntime(),
): Promise<string> {
  return digest(clubAssemblyIdentityPayload(assembly), runtime);
}
