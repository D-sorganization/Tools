/** Sidecar records unlocked only by a validated ClubAssembly binding. */

import { type ClubAssemblyBinding } from "./clubAssemblyBinding";

export const BINDING_AUTHORITY_LIMITATION =
  "The imported source-authority declaration is preserved but is not independently certified by this application.";

export function boundCapabilityContract(): Record<string, unknown> {
  return {
    assembly_mass_properties: { status: "available" },
    head_center_of_mass: { status: "available" },
    head_full_inertia_tensor: { status: "available" },
    head_mass: { status: "available" },
    mesh_identity: { status: "available" },
    world_from_head_attitude: {
      missing: ["complete world-from-head rotation"],
      status: "unavailable",
    },
  };
}

export function boundFrameContract(
  binding: ClubAssemblyBinding,
  baseFrames: Record<string, unknown>,
): Record<string, unknown> {
  const transform = binding.headBinding.headComponentFromSelectedHead;
  return {
    ...baseFrames,
    head_component_from_head: {
      from_frame_id: transform.from_frame_id,
      rotation: transform.rotation,
      status: "available",
      to_frame_id: transform.to_frame_id,
      translation_m: transform.translation_m,
    },
    assembly: {
      frame_id: binding.assembly.frame_id,
      length_unit: "m",
      status: "available",
    },
  };
}

export function boundMassProperties(
  binding: ClubAssemblyBinding,
): Record<string, unknown> {
  const head = binding.headPropertiesInSelectedFrame;
  const assembly = binding.assemblyMassProperties;
  const provenance = "validated_club_assembly_binding";
  return {
    assembly: {
      center_of_mass_m: assembly.centerOfMassM,
      component_ids: assembly.componentIds,
      frame_id: assembly.frameId,
      inertia_tensor_at_com_kg_m2: assembly.inertiaAtComKgM2,
      provenance,
      status: "available",
      total_mass_kg: assembly.totalMassKg,
    },
    head: {
      center_of_mass_m: {
        frame_id: head.frame_id,
        provenance,
        status: "available",
        value: head.center_of_mass_m,
      },
      inertia_tensor_at_com_kg_m2: {
        about: "head_center_of_mass",
        frame_id: head.frame_id,
        provenance,
        status: "available",
        value: head.inertia_at_com_kg_m2,
      },
      mass_kg: {
        provenance,
        status: "available",
        value: head.mass_kg,
      },
    },
  };
}

export function bindingProvenance(
  binding: ClubAssemblyBinding,
): Record<string, unknown> {
  const authority = binding.sourceAuthority;
  return {
    assembly_id: binding.assemblyIdentity.assemblyId,
    assembly_sha256: binding.assemblyIdentity.sha256,
    binding_format: binding.format,
    head_component_id: binding.headBinding.headComponentId,
    selected_spec_sha256: binding.selectedSpecIdentity.sha256,
    source_authority: {
      kind: authority.kind,
      authority_id: authority.authorityId,
      document_id: authority.documentId,
      revision: authority.revision,
    },
  };
}
