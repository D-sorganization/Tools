/** Club fitting document, report, and heavy-hit coupling model layer (C7, H4). */

export const FITTING_DOCUMENT_FORMAT = "golf_club.fitting_document/1" as const;
export const FITTING_REPORT_FORMAT = "golf_club.fitting_report/1" as const;
export const IMPACT_COUPLING_REPORT_FORMAT = "golf_club.impact_coupling_report/1" as const;
export const BODY_CHAIN_FORMAT = "swing_sim.body_chain/1" as const;

export interface FaceGeometry {
  readonly loft_deg: number;
  readonly lie_deg: number;
  readonly bulge_m: number;
  readonly roll_m: number;
}

export interface MeshReference {
  readonly name: string;
  readonly sha256: string;
  readonly density_kg_m3?: number;
  readonly target_mass_kg?: number;
}

export interface FittingProvenance {
  readonly source_kind: "oem_export" | "measured" | "parametric" | "cad_derived";
  readonly tool_name: string;
  readonly exported_at: string;
}

export interface ShaftTipMass {
  readonly mass_kg: number;
  readonly cg_back_m: number;
  readonly cg_toe_m: number;
  readonly cg_drop_m: number;
}

export interface ClubFittingDocument {
  readonly format: typeof FITTING_DOCUMENT_FORMAT;
  readonly document_id: string;
  readonly face: FaceGeometry;
  readonly assembly: Record<string, unknown>;
  readonly shaft_profile: Record<string, unknown>;
  readonly tip_mass: ShaftTipMass;
  readonly provenance: FittingProvenance;
  readonly mesh_reference?: MeshReference;
}

export interface ShaftDeliveryDeltas {
  readonly dynamic_loft_add_deg: number;
  readonly face_closure_deg: number;
  readonly lie_toe_down_deg: number;
  readonly kick_speed_mps: number;
  readonly first_mode_hz: number;
  readonly model_name: string;
}

export interface ClubOutcome {
  readonly label: string;
  readonly delivered_loft_deg: number;
  readonly face_angle_deg: number;
  readonly lie_toe_down_deg: number;
  readonly clubhead_speed_mps: number;
  readonly ball_speed_mps: number;
  readonly launch_angle_deg: number;
  readonly backspin_rpm: number;
  readonly carry_m: number;
  readonly max_height_m: number;
  readonly flight_time_s: number;
  readonly lateral_m: number;
  readonly shaft: ShaftDeliveryDeltas;
  readonly deltas_vs_baseline?: {
    readonly carry_m: number;
    readonly ball_speed_mps: number;
    readonly launch_angle_deg: number;
    readonly backspin_rpm: number;
    readonly lateral_m: number;
  };
}

export interface GripKinematics {
  readonly omega_rad_s: number;
  readonly alpha_rad_s2: number;
  readonly swing_radius_m: number;
  readonly downswing_duration_s: number;
  readonly release_recovery: number;
}

export interface CounterfactualSpec {
  readonly label: string;
  readonly head_mass_scale: number;
  readonly cg_back_delta_m: number;
  readonly cg_toe_delta_m: number;
  readonly loft_delta_deg: number;
  readonly ei_scale: number;
  readonly gj_scale: number;
}

export interface FittingReport {
  readonly format: typeof FITTING_REPORT_FORMAT;
  readonly document_id: string;
  readonly grip: GripKinematics;
  readonly baseline: ClubOutcome;
  readonly counterfactuals: readonly ClubOutcome[];
}

export interface GripBoundary {
  readonly effective_mass_kg: number;
  readonly stiffness_n_m: number;
  readonly damping_n_s_m: number;
  readonly provenance: string;
}

export interface CoupledImpactResult {
  readonly ball_speed_mps: number;
  readonly free_head_ball_speed_mps: number;
  readonly decoupling_fraction: number;
  readonly contact_time_s: number;
  readonly peak_contact_force_n: number;
  readonly grip_provenance: string;
  readonly axis?: string;
  readonly value?: number;
}

export interface ImpactCouplingReport {
  readonly format: typeof IMPACT_COUPLING_REPORT_FORMAT;
  readonly baseline: CoupledImpactResult;
  readonly counterfactuals: readonly CoupledImpactResult[];
}

export interface ChainJoint {
  readonly name: string;
  readonly type: "revolute" | "prismatic" | "ball" | "free" | "fixed";
  readonly stiffness: number;
  readonly damping: number;
}

export interface ChainBody {
  readonly name: string;
  readonly mass_kg: number;
  readonly inertia_diag_kg_m2: readonly [number, number, number];
  readonly parent: string | null;
  readonly joint: ChainJoint | null;
}

export interface BodyChain {
  readonly format: typeof BODY_CHAIN_FORMAT;
  readonly source_id: string;
  readonly bodies: readonly ChainBody[];
}

function requireObject(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireFinite(value: unknown, name: string, min?: number, max?: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  if (min !== undefined && value < min) {
    throw new RangeError(`${name} must be >= ${min}`);
  }
  if (max !== undefined && value > max) {
    throw new RangeError(`${name} must be <= ${max}`);
  }
  return value;
}

function requireString(value: unknown, name: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${name} must be a nonempty string`);
  }
  return value;
}

export function parseFittingDocument(text: string): ClubFittingDocument {
  const raw = requireObject(JSON.parse(text), "fitting document");
  if (raw.format !== FITTING_DOCUMENT_FORMAT) {
    throw new ValueError(`Expected format ${FITTING_DOCUMENT_FORMAT}, got ${String(raw.format)}`);
  }
  const documentId = requireString(raw.document_id, "document_id");
  const faceRaw = requireObject(raw.face, "face");
  const face: FaceGeometry = {
    loft_deg: requireFinite(faceRaw.loft_deg, "loft_deg", 0, 90),
    lie_deg: requireFinite(faceRaw.lie_deg, "lie_deg", 30, 90),
    bulge_m: requireFinite(faceRaw.bulge_m ?? 0, "bulge_m", 0),
    roll_m: requireFinite(faceRaw.roll_m ?? 0, "roll_m", 0),
  };
  const tipRaw = requireObject(raw.tip_mass, "tip_mass");
  const tipMass: ShaftTipMass = {
    mass_kg: requireFinite(tipRaw.mass_kg, "mass_kg", 0),
    cg_back_m: requireFinite(tipRaw.cg_back_m, "cg_back_m"),
    cg_toe_m: requireFinite(tipRaw.cg_toe_m, "cg_toe_m"),
    cg_drop_m: requireFinite(tipRaw.cg_drop_m, "cg_drop_m"),
  };
  const provRaw = requireObject(raw.provenance, "provenance");
  const provenance: FittingProvenance = {
    source_kind: requireString(provRaw.source_kind, "source_kind") as FittingProvenance["source_kind"],
    tool_name: requireString(provRaw.tool_name, "tool_name"),
    exported_at: requireString(provRaw.exported_at, "exported_at"),
  };
  let meshRef: MeshReference | undefined;
  if (raw.mesh_reference !== undefined) {
    const mRaw = requireObject(raw.mesh_reference, "mesh_reference");
    meshRef = {
      name: requireString(mRaw.name, "mesh_reference.name"),
      sha256: requireString(mRaw.sha256, "mesh_reference.sha256"),
      density_kg_m3: mRaw.density_kg_m3 !== undefined ? requireFinite(mRaw.density_kg_m3, "density_kg_m3", 0) : undefined,
      target_mass_kg: mRaw.target_mass_kg !== undefined ? requireFinite(mRaw.target_mass_kg, "target_mass_kg", 0) : undefined,
    };
  }
  return {
    format: FITTING_DOCUMENT_FORMAT,
    document_id: documentId,
    face,
    assembly: requireObject(raw.assembly, "assembly"),
    shaft_profile: requireObject(raw.shaft_profile, "shaft_profile"),
    tip_mass: tipMass,
    provenance,
    mesh_reference: meshRef,
  };
}

export function serializeFittingDocument(doc: ClubFittingDocument): string {
  return JSON.stringify(doc);
}

export function parseFittingReport(text: string): FittingReport {
  const raw = requireObject(JSON.parse(text), "fitting report");
  if (raw.format !== FITTING_REPORT_FORMAT) {
    throw new ValueError(`Expected format ${FITTING_REPORT_FORMAT}, got ${String(raw.format)}`);
  }
  const documentId = requireString(raw.document_id, "document_id");
  const gripRaw = requireObject(raw.grip, "grip");
  const grip: GripKinematics = {
    omega_rad_s: requireFinite(gripRaw.omega_rad_s, "omega_rad_s"),
    alpha_rad_s2: requireFinite(gripRaw.alpha_rad_s2, "alpha_rad_s2"),
    swing_radius_m: requireFinite(gripRaw.swing_radius_m, "swing_radius_m", 0),
    downswing_duration_s: requireFinite(gripRaw.downswing_duration_s, "downswing_duration_s", 0),
    release_recovery: requireFinite(gripRaw.release_recovery, "release_recovery", 0, 1),
  };

  const parseOutcome = (oRaw: Record<string, unknown>): ClubOutcome => {
    const sRaw = requireObject(oRaw.shaft, "shaft");
    return {
      label: requireString(oRaw.label, "label"),
      delivered_loft_deg: requireFinite(oRaw.delivered_loft_deg, "delivered_loft_deg"),
      face_angle_deg: requireFinite(oRaw.face_angle_deg, "face_angle_deg"),
      lie_toe_down_deg: requireFinite(oRaw.lie_toe_down_deg, "lie_toe_down_deg"),
      clubhead_speed_mps: requireFinite(oRaw.clubhead_speed_mps, "clubhead_speed_mps", 0),
      ball_speed_mps: requireFinite(oRaw.ball_speed_mps, "ball_speed_mps", 0),
      launch_angle_deg: requireFinite(oRaw.launch_angle_deg, "launch_angle_deg"),
      backspin_rpm: requireFinite(oRaw.backspin_rpm, "backspin_rpm"),
      carry_m: requireFinite(oRaw.carry_m, "carry_m"),
      max_height_m: requireFinite(oRaw.max_height_m, "max_height_m"),
      flight_time_s: requireFinite(oRaw.flight_time_s, "flight_time_s", 0),
      lateral_m: requireFinite(oRaw.lateral_m, "lateral_m"),
      shaft: {
        dynamic_loft_add_deg: requireFinite(sRaw.dynamic_loft_add_deg, "dynamic_loft_add_deg"),
        face_closure_deg: requireFinite(sRaw.face_closure_deg, "face_closure_deg"),
        lie_toe_down_deg: requireFinite(sRaw.lie_toe_down_deg, "lie_toe_down_deg"),
        kick_speed_mps: requireFinite(sRaw.kick_speed_mps, "kick_speed_mps"),
        first_mode_hz: requireFinite(sRaw.first_mode_hz, "first_mode_hz", 0),
        model_name: requireString(sRaw.model_name, "model_name"),
      },
      deltas_vs_baseline: oRaw.deltas_vs_baseline ? (oRaw.deltas_vs_baseline as ClubOutcome["deltas_vs_baseline"]) : undefined,
    };
  };

  const baseline = parseOutcome(requireObject(raw.baseline, "baseline"));
  if (!Array.isArray(raw.counterfactuals)) {
    throw new TypeError("counterfactuals must be an array");
  }
  const counterfactuals = raw.counterfactuals.map((c) => parseOutcome(requireObject(c, "counterfactual")));

  return {
    format: FITTING_REPORT_FORMAT,
    document_id: documentId,
    grip,
    baseline,
    counterfactuals,
  };
}

export function parseImpactCouplingReport(text: string): ImpactCouplingReport {
  const raw = requireObject(JSON.parse(text), "impact coupling report");
  if (raw.format !== IMPACT_COUPLING_REPORT_FORMAT) {
    throw new ValueError(`Expected format ${IMPACT_COUPLING_REPORT_FORMAT}, got ${String(raw.format)}`);
  }
  const parseResult = (rRaw: Record<string, unknown>): CoupledImpactResult => ({
    ball_speed_mps: requireFinite(rRaw.ball_speed_mps, "ball_speed_mps", 0),
    free_head_ball_speed_mps: requireFinite(rRaw.free_head_ball_speed_mps, "free_head_ball_speed_mps", 0),
    decoupling_fraction: requireFinite(rRaw.decoupling_fraction, "decoupling_fraction", 0, 1),
    contact_time_s: requireFinite(rRaw.contact_time_s, "contact_time_s", 0),
    peak_contact_force_n: requireFinite(rRaw.peak_contact_force_n, "peak_contact_force_n", 0),
    grip_provenance: requireString(rRaw.grip_provenance, "grip_provenance"),
    axis: typeof rRaw.axis === "string" ? rRaw.axis : undefined,
    value: typeof rRaw.value === "number" ? rRaw.value : undefined,
  });

  const baseline = parseResult(requireObject(raw.baseline, "baseline"));
  if (!Array.isArray(raw.counterfactuals)) {
    throw new TypeError("counterfactuals must be an array");
  }
  const counterfactuals = raw.counterfactuals.map((c) => parseResult(requireObject(c, "counterfactual")));
  return {
    format: IMPACT_COUPLING_REPORT_FORMAT,
    baseline,
    counterfactuals,
  };
}

export function parseBodyChain(text: string): BodyChain {
  const raw = requireObject(JSON.parse(text), "body chain");
  if (raw.format !== BODY_CHAIN_FORMAT) {
    throw new ValueError(`Expected format ${BODY_CHAIN_FORMAT}, got ${String(raw.format)}`);
  }
  const sourceId = requireString(raw.source_id, "source_id");
  if (!Array.isArray(raw.bodies) || raw.bodies.length === 0) {
    throw new TypeError("bodies must be a nonempty array");
  }
  const bodies: ChainBody[] = raw.bodies.map((bRawUnknown) => {
    const bRaw = requireObject(bRawUnknown, "body");
    const name = requireString(bRaw.name, "body.name");
    const massKg = requireFinite(bRaw.mass_kg, "body.mass_kg", 0);
    if (!Array.isArray(bRaw.inertia_diag_kg_m2) || bRaw.inertia_diag_kg_m2.length !== 3) {
      throw new TypeError("body.inertia_diag_kg_m2 must have 3 elements");
    }
    const inertia = [
      requireFinite(bRaw.inertia_diag_kg_m2[0], "inertia[0]", 0),
      requireFinite(bRaw.inertia_diag_kg_m2[1], "inertia[1]", 0),
      requireFinite(bRaw.inertia_diag_kg_m2[2], "inertia[2]", 0),
    ] as const;
    let joint: ChainJoint | null = null;
    if (bRaw.joint !== null && bRaw.joint !== undefined) {
      const jRaw = requireObject(bRaw.joint, "body.joint");
      joint = {
        name: requireString(jRaw.name, "joint.name"),
        type: requireString(jRaw.type, "joint.type") as ChainJoint["type"],
        stiffness: requireFinite(jRaw.stiffness ?? 0, "joint.stiffness", 0),
        damping: requireFinite(jRaw.damping ?? 0, "joint.damping", 0),
      };
    }
    return {
      name,
      mass_kg: massKg,
      inertia_diag_kg_m2: inertia,
      parent: typeof bRaw.parent === "string" ? bRaw.parent : null,
      joint,
    };
  });
  return {
    format: BODY_CHAIN_FORMAT,
    source_id: sourceId,
    bodies,
  };
}

export function gripBoundaryReduction(
  chain: BodyChain,
  options: {
    readonly handBodies: readonly string[];
    readonly boundaryJointOf: string;
    readonly stiffnessOverride?: number;
    readonly dampingOverride?: number;
  },
): GripBoundary {
  if (!options.handBodies || options.handBodies.length === 0) {
    throw new ValueError("handBodies must be nonempty");
  }
  const bodyMap = new Map<string, ChainBody>();
  chain.bodies.forEach((b) => bodyMap.set(b.name, b));

  let totalMass = 0;
  for (const name of options.handBodies) {
    const body = bodyMap.get(name);
    if (!body) {
      throw new ValueError(`Unknown hand body: ${name}`);
    }
    totalMass += body.mass_kg;
  }
  if (totalMass <= 0) {
    throw new ValueError("Total hand mass must be > 0");
  }

  const boundaryBody = bodyMap.get(options.boundaryJointOf);
  if (!boundaryBody) {
    throw new ValueError(`Unknown boundary body: ${options.boundaryJointOf}`);
  }
  if (!boundaryBody.joint) {
    throw new ValueError(`Boundary body ${options.boundaryJointOf} has no joint`);
  }

  const stiffness = options.stiffnessOverride !== undefined ? requireFinite(options.stiffnessOverride, "stiffnessOverride", 0) : boundaryBody.joint.stiffness;
  const damping = options.dampingOverride !== undefined ? requireFinite(options.dampingOverride, "dampingOverride", 0) : boundaryBody.joint.damping;

  const overrideNote = options.stiffnessOverride !== undefined || options.dampingOverride !== undefined ? " (stiffness/damping overridden by caller)" : "";
  const provenance = `${chain.source_id}: bodies [${options.handBodies.join(", ")}] via joint '${boundaryBody.joint.name}'${overrideNote}`;

  return {
    effective_mass_kg: totalMass,
    stiffness_n_m: stiffness,
    damping_n_s_m: damping,
    provenance,
  };
}

class ValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValueError";
  }
}
