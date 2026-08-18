import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import { impactKinematics } from "../model/impactKinematics";
import type { SimulationRunTs } from "../model/simulation";

interface Props {
  run: SimulationRunTs;
  scenario: ImpactScenario;
  club: ClubSpec;
}

const number = (value: number | null, unit: string, decimals = 2) =>
  value === null ? "Unavailable" : `${value.toFixed(decimals)} ${unit}`;

export function ImpactKinematicsPanel({ run, scenario, club }: Props) {
  const metrics = impactKinematics(run, scenario, club);
  const dplane = metrics.faceCenterDPlane;
  const entries = [
    { label: "Face-Center Club Path", value: number(dplane.clubPathDeg, "°"),
      equation: "atan2(v_face_center · right, v_face_center · target)", detail: "Horizontal heading of the rigid-body face-center velocity; positive is right/in-to-out in the app convention." },
    { label: "Face-Center Attack Angle", value: number(dplane.attackAngleDeg, "°"),
      equation: "atan2(v_face_center · up, |v_horizontal|)", detail: "Positive is ascending and negative is descending in the app frame." },
    { label: "Face Angle", value: number(dplane.faceAngleDeg, "°"),
      equation: "atan2(n_face · right, n_face · target)", detail: "Horizontal heading of the face-center normal; positive points right/open in the app convention." },
    { label: "Dynamic Loft", value: number(dplane.dynamicLoftDeg, "°"),
      equation: "atan2(n_face · up, |n_face,horizontal|)", detail: "Vertical elevation of the delivered face-center normal." },
    { label: "Face to Path", value: number(dplane.faceToPathDeg, "°"),
      equation: "wrap(face_angle − club_path)", detail: "Positive means the face points right/open relative to the face-center path." },
    { label: "Spin Loft (Exact 3D)", value: number(dplane.spinLoft3dDeg, "°"),
      equation: "acos(unit(v_face_center) · n_face_center)", detail: "Coordinate-free included angle that defines the displayed shaded sector." },
    { label: "Spin Loft (Planar Approximation)", value: number(dplane.planarSpinLoftDeg, "°"),
      equation: "|dynamic_loft − attack_angle|", detail: "Two-dimensional approximation; use the reported residual to assess its error." },
    { label: "3D Minus Planar Residual", value: number(dplane.spinLoftResidualDeg, "°"),
      equation: "spin_loft_3D − |dynamic_loft − attack_angle|", detail: "Difference created by the full horizontal and vertical geometry." },
    { label: "D-Plane Normal Tilt", value: number(dplane.dplaneTiltDeg, "°"),
      equation: "atan2(−n_D · up, |n_D,horizontal|)", detail: "Positive is face-right; that is fade-side only under the current right-handed display convention. Geometry alone does not predict curvature. Unavailable for a degenerate D-plane." },
    { label: "D-Plane Inclination", value: number(dplane.dplaneInclinationDeg, "°"),
      equation: "acos(|n_D · up|)", detail: "Unsigned angle between the D-plane and the ground plane." },
    { label: "Reference-Point AoA", value: number(metrics.referenceAoaDeg, "°"),
      equation: "AoA(v_axis)", detail: "Signed descent angle of the physical shaft-axis datum relative to the ground plane." },
    { label: "Contact-Point AoA", value: number(metrics.contactAoaDeg, "°"),
      equation: "atan2(v_contact · up, |v_horizontal|)", detail: "Signed attack angle at the declared face contact point. Negative values descend toward the ground." },
    { label: "Without Shaft Rotation", value: number(metrics.withoutShaftAoaDeg, "°"),
      equation: "AoA(v_contact − v_shaft)", detail: "Rigid-body counterfactual with only the angular-velocity component parallel to the shaft removed." },
    { label: "Shaft AoA Contribution", value: number(metrics.shaftAoaContributionDeg, "°"),
      equation: "AoA(v_contact) − AoA(v_contact − v_shaft)", detail: "Non-additive counterfactual delta; this is not an Euler-angle decomposition." },
    { label: "Shaft Rotation Rate", value: number(metrics.shaftRotationRateDps, "°/s", 1),
      equation: "ω · ŝ", detail: "Signed projection of rigid-head angular velocity onto the declared physical shaft axis." },
    { label: "Shaft-Induced Vertical Velocity", value: number(metrics.shaftVerticalVelocityMps, "m/s", 3),
      equation: "(ω_shaft × r_contact/shaft) · up", detail: "Vertical contact-point speed created by rotation about the shaft datum." },
    { label: "Shaft Share of Vertical Velocity", value: number(metrics.shaftVerticalVelocityShare, "×", 3),
      equation: "v_shaft,vertical / v_contact,vertical", detail: "Dimensionless signed share. It is unavailable when total vertical speed is zero." },
    { label: "Face-Normal 3D Rate", value: number(metrics.faceNormalRateDps, "°/s", 1),
      equation: "|ω × n_face|", detail: "Coordinate-free angular rate of the face normal in the inertial app frame." },
    { label: "Leading-Edge 3D Rate", value: number(metrics.leadingEdgeRateDps, "°/s", 1),
      equation: "|ω × e_leading|", detail: "Coordinate-free angular rate of the leading-edge direction in the inertial app frame." },
  ];
  return (
    <aside aria-label="Impact Kinematics Engineering Readout"
      className="mb-3 rounded-lg border border-cyan-400/30 bg-cyan-950/10 p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-semibold text-cyan-200">{metrics.eventLabel} Kinematics</h3>
        <span className="text-xs tabular-nums text-cyan-300/80">
          {metrics.eventTimeS.toFixed(3)} s · app frame: x target, y up, z right
        </span>
      </div>
      <div className="grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        {entries.map((entry) => <details key={entry.label}
          className="group rounded border border-slate-700/70 bg-slate-900/60 p-2 transition-colors open:border-cyan-400/50 hover:border-slate-500">
          <summary className="cursor-pointer list-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400">
            <div className="flex items-center justify-between gap-2 text-xs text-slate-400">
              {entry.label}
              <span aria-hidden="true" className="text-cyan-400 transition-transform group-open:rotate-90">›</span>
            </div>
            <p className="font-mono text-slate-100">{entry.value}</p>
            <span className="text-[10px] font-medium uppercase tracking-wide text-cyan-400/80">Click for Definition</span>
          </summary>
          <div className="mt-2 border-t border-slate-700 pt-2 text-xs leading-relaxed text-slate-300">
            <p><b>Equation:</b> <code>{entry.equation}</code></p>
            <p className="mt-1"><b>Frame:</b> app frame, x target, y up, z right.</p>
            <p className="mt-1"><b>Assumptions:</b> {entry.detail}</p>
          </div>
        </details>)}
      </div>
      <p className="mt-2 text-xs text-slate-400">
        <b className="text-slate-300">Geometry Basis:</b> {metrics.geometryBasis}.{" "}
        <b className="text-slate-300">Model Boundary:</b> {metrics.modelLimitations}
        {" "}<b className="text-slate-300">D-Plane State:</b> {dplane.status.split("_").join(" ")}. Geometry alone does not predict launch or ball spin without the declared collision model.
      </p>
    </aside>
  );
}
