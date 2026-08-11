/** Strict selected ClubSpec to shared ClubAssembly import/binding contract. */

import { type ClubSpec } from "./club";
import {
  assembleMassProperties,
  headPropertiesInSelectedFrame,
  requirePositiveInertia,
  validateClubAssembly,
  validateRigidTransform,
  type AssembledMassPropertiesRecord,
  type ClubAssemblyRecord,
  type ClubComponentRecord,
  type ComponentMassPropertiesRecord,
  type RigidTransformRecord,
} from "./clubAssemblyMassProperties";
import {
  browserBindingDigestRuntime,
  canonicalIdentityPayload,
  clubAssemblyIdentity,
  clubSpecIdentity,
  selectedSpecSnapshot,
  type ClubAssemblyBindingDigestRuntime,
} from "./clubAssemblyBindingIdentity";
import { parseStrictJson } from "./strictJsonObject";

export type { ClubAssemblyBindingDigestRuntime };

export const CLUB_ASSEMBLY_BINDING_FORMAT =
  "rate_of_closure.club_assembly_binding/1";
export const CLUB_SPEC_IDENTITY_FORMAT = "rate_of_closure.club_spec_identity/1";
export const MAX_BINDING_BYTES = 4 * 1024 * 1024;

export type MassPropertyAuthorityKind =
  "measured" | "manufacturer" | "cad_integrated" | "qualified_analysis";

export interface ClubAssemblySourceAuthority {
  kind: MassPropertyAuthorityKind;
  authorityId: string;
  documentId: string;
  revision: string;
}

export interface ClubAssemblyBinding {
  format: typeof CLUB_ASSEMBLY_BINDING_FORMAT;
  selectedSpecIdentity: {
    format: typeof CLUB_SPEC_IDENTITY_FORMAT;
    sha256: string;
    snapshot: Record<string, unknown>;
  };
  assemblyIdentity: {
    format: "golf_club.assembly/1";
    assemblyId: string;
    sha256: string;
  };
  sourceAuthority: ClubAssemblySourceAuthority;
  headBinding: {
    headComponentId: string;
    headComponentFromSelectedHead: RigidTransformRecord;
  };
  assembly: ClubAssemblyRecord;
  assemblyMassProperties: AssembledMassPropertiesRecord;
  headPropertiesInSelectedFrame: ComponentMassPropertiesRecord;
}

const AUTHORITY_KINDS = new Set<MassPropertyAuthorityKind>([
  "measured",
  "manufacturer",
  "cad_integrated",
  "qualified_analysis",
]);
const SHA256_HEX = /^[a-f0-9]{64}$/u;
const UNITS = {
  angle: "degree",
  inertia: "kg_m2",
  length: "m",
  mass: "kg",
};
const SPEC_FIELDS = [
  "name",
  "club_type",
  "length_m",
  "head_mass_kg",
  "loft_deg",
  "lie_deg",
  "moi_about_shaft_kg_m2",
  "cg_depth_m",
  "cg_height_m",
  "face_bulge_radius_m",
  "face_roll_radius_m",
  "head_style",
] as const;
const HEAD_MASS_TOLERANCE_KG = 1e-12;

function record(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be a JSON object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    throw new Error(`${name} fields do not match schema`);
  }
}

function identifier(value: unknown, name: string): string {
  if (typeof value !== "string" || !value || value.trim() !== value) {
    throw new Error(`${name} must be a nonempty trimmed string`);
  }
  return value;
}

function sha256(value: unknown, name: string): string {
  if (typeof value !== "string" || !SHA256_HEX.test(value)) {
    throw new Error(`${name} must be lowercase SHA-256`);
  }
  return value;
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  return (
    left.byteLength === right.byteLength &&
    left.every((value, index) => value === right[index])
  );
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

function parseAuthority(value: unknown): ClubAssemblySourceAuthority {
  const data = record(value, "source_authority");
  exactKeys(
    data,
    ["kind", "authority_id", "document_id", "revision"],
    "source_authority",
  );
  const kind = identifier(
    data.kind,
    "authority kind",
  ) as MassPropertyAuthorityKind;
  if (!AUTHORITY_KINDS.has(kind)) {
    throw new Error(`unsupported authority kind ${kind}`);
  }
  return deepFreeze({
    kind,
    authorityId: identifier(data.authority_id, "authority_id"),
    documentId: identifier(data.document_id, "document_id"),
    revision: identifier(data.revision, "revision"),
  });
}

function parseSelectedIdentity(value: unknown): {
  format: typeof CLUB_SPEC_IDENTITY_FORMAT;
  sha256: string;
  snapshot: Record<string, unknown>;
} {
  const data = record(value, "selected_spec_identity");
  exactKeys(data, ["format", "sha256", "snapshot"], "selected_spec_identity");
  if (data.format !== CLUB_SPEC_IDENTITY_FORMAT) {
    throw new Error("selected ClubSpec identity format is unsupported");
  }
  const snapshot = record(data.snapshot, "selected spec snapshot");
  exactKeys(snapshot, SPEC_FIELDS, "selected spec snapshot");
  return {
    format: CLUB_SPEC_IDENTITY_FORMAT,
    sha256: sha256(data.sha256, "selected ClubSpec identity"),
    snapshot,
  };
}

function parseAssemblyIdentity(value: unknown): {
  format: "golf_club.assembly/1";
  assemblyId: string;
  sha256: string;
} {
  const data = record(value, "assembly_identity");
  exactKeys(data, ["format", "assembly_id", "sha256"], "assembly_identity");
  if (data.format !== "golf_club.assembly/1") {
    throw new Error("assembly identity format is unsupported");
  }
  return {
    format: "golf_club.assembly/1",
    assemblyId: identifier(data.assembly_id, "assembly identity assembly_id"),
    sha256: sha256(data.sha256, "assembly identity"),
  };
}

function parseHeadBinding(value: unknown): ClubAssemblyBinding["headBinding"] {
  const data = record(value, "head_binding");
  exactKeys(
    data,
    ["head_component_id", "head_component_from_selected_head"],
    "head_binding",
  );
  return {
    headComponentId: identifier(data.head_component_id, "head component"),
    headComponentFromSelectedHead: validateRigidTransform(
      data.head_component_from_selected_head,
    ),
  };
}

function validateUnits(value: unknown): void {
  const data = record(value, "units");
  exactKeys(data, Object.keys(UNITS), "units");
  if (Object.entries(UNITS).some(([key, expected]) => data[key] !== expected)) {
    throw new Error("binding units must use the declared SI contract");
  }
}

function validateHeadRelationship(
  spec: ClubSpec,
  assembly: ClubAssemblyRecord,
  headBinding: ClubAssemblyBinding["headBinding"],
): ClubComponentRecord {
  const heads = assembly.components.filter(
    (component) => component.mass_properties.role === "head",
  );
  if (heads.length !== 1)
    throw new Error("assembly must contain exactly one head component");
  const head = heads[0];
  if (head.mass_properties.component_id !== headBinding.headComponentId) {
    throw new Error(
      "head component identifier does not select the unique head",
    );
  }
  const transform = headBinding.headComponentFromSelectedHead;
  if (transform.from_frame_id !== "rate_of_closure.head") {
    throw new Error("head transform must start in the selected head frame");
  }
  if (transform.to_frame_id !== head.mass_properties.frame_id) {
    throw new Error(
      "head component frame does not match the binding transform",
    );
  }
  if (
    Math.abs(head.mass_properties.mass_kg - spec.headMassKg) >
    HEAD_MASS_TOLERANCE_KG
  ) {
    throw new Error(
      "bound head mass does not match selected ClubSpec head mass",
    );
  }
  requirePositiveInertia(
    head.mass_properties.inertia_at_com_kg_m2,
    "head inertia",
  );
  return head;
}

function parseRoot(text: string): Record<string, unknown> {
  if (new TextEncoder().encode(text).byteLength > MAX_BINDING_BYTES) {
    throw new Error("club assembly binding exceeds the 4 MiB limit");
  }
  const data = record(parseStrictJson(text), "binding");
  exactKeys(
    data,
    [
      "format",
      "selected_spec_identity",
      "assembly_identity",
      "source_authority",
      "units",
      "head_binding",
      "assembly",
    ],
    "binding",
  );
  if (data.format !== CLUB_ASSEMBLY_BINDING_FORMAT) {
    throw new Error("club assembly binding format is unsupported");
  }
  return data;
}

/** Parse and cryptographically validate one binding against the current selection. */
export async function parseClubAssemblyBinding(
  spec: ClubSpec,
  text: string,
  runtime: ClubAssemblyBindingDigestRuntime = browserBindingDigestRuntime(),
): Promise<ClubAssemblyBinding> {
  const data = parseRoot(text);
  validateUnits(data.units);
  const selectedSpecIdentity = parseSelectedIdentity(
    data.selected_spec_identity,
  );
  const expectedSnapshot = selectedSpecSnapshot(spec);
  const expectedSpecDigest = await clubSpecIdentity(spec, runtime);
  if (
    !sameBytes(
      canonicalIdentityPayload(selectedSpecIdentity.snapshot),
      canonicalIdentityPayload(expectedSnapshot),
    ) ||
    selectedSpecIdentity.sha256 !== expectedSpecDigest
  ) {
    throw new Error("binding selected ClubSpec identity does not match");
  }
  const assembly = validateClubAssembly(data.assembly);
  const assemblyIdentity = parseAssemblyIdentity(data.assembly_identity);
  const expectedAssemblyDigest = await clubAssemblyIdentity(assembly, runtime);
  if (
    assemblyIdentity.assemblyId !== assembly.assembly_id ||
    assemblyIdentity.sha256 !== expectedAssemblyDigest
  ) {
    throw new Error("assembly identity does not match embedded assembly");
  }
  const sourceAuthority = parseAuthority(data.source_authority);
  const headBinding = parseHeadBinding(data.head_binding);
  const head = validateHeadRelationship(spec, assembly, headBinding);
  const assemblyMassProperties = assembleMassProperties(assembly);
  return deepFreeze({
    format: CLUB_ASSEMBLY_BINDING_FORMAT,
    selectedSpecIdentity,
    assemblyIdentity,
    sourceAuthority,
    headBinding,
    assembly,
    assemblyMassProperties,
    headPropertiesInSelectedFrame: headPropertiesInSelectedFrame(
      head,
      headBinding.headComponentFromSelectedHead,
    ),
  });
}

/** Revalidate a retained binding after any selected-spec UI change. */
export function assertBindingMatchesSpec(
  binding: ClubAssemblyBinding,
  spec: ClubSpec,
): void {
  if (
    !sameBytes(
      canonicalIdentityPayload(binding.selectedSpecIdentity.snapshot),
      canonicalIdentityPayload(selectedSpecSnapshot(spec)),
    )
  ) {
    throw new Error("binding selected ClubSpec identity does not match");
  }
}
