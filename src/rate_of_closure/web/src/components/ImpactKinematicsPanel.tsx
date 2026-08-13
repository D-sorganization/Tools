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

const vector = (value: readonly number[], unit: string) =>
  `[${value.map((component) => component.toFixed(4)).join(", ")}] ${unit}`;

export function ImpactKinematicsPanel({ run, scenario, club }: Props) {
  const metrics = impactKinematics(run, scenario, club);
  const dplane = metrics.faceCenterDPlane;
  const sasho = metrics.sashoFaceCenterRotation;
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
    { label: "Shaft-Rotation Shapley AoA", value: number(metrics.shaftShapleyAoaDeg, "°"),
      equation: "mean marginal AoA across both factor orders", detail: "Order-independent two-factor attribution about shaft-axis translation; it remains a model attribution, not an independently measured cause." },
    { label: "Sasho Face-Center Rotation-Only AoA", value: number(metrics.sashoFaceCenterRotation.aoaDeg, "°"),
      equation: "AoA(ω × (F − nearest_shaft(F)))", detail: `Method ${metrics.sashoFaceCenterRotation.methodId}. Uses complete club angular velocity and the nearest point on the physical shaft line. It is descriptive and is not interchangeable with shaft-axis-only or Shapley attribution.` },
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
      <p className="mt-2 rounded border border-cyan-400/20 bg-slate-950/40 p-2 text-xs text-slate-300">
        <b>AoA method options:</b> compare the remove-shaft counterfactual,
        two-factor Shapley attribution, and Sasho nearest-shaft face-center
        rotation-only AoA. They answer different questions and are not additive.
      </p>
      <dl aria-label="Sasho nearest-shaft geometry"
        className="mt-2 grid gap-x-4 gap-y-1 rounded border border-teal-400/20 bg-teal-950/10 p-2 text-xs sm:grid-cols-2">
        <div><dt className="text-slate-400">Method ID</dt><dd>{sasho.methodId}</dd></div>
        <div><dt className="text-slate-400">Nearest shaft point Q</dt><dd>{vector(sasho.nearestShaftPointM, "m")}</dd></div>
        <div><dt className="text-slate-400">Perpendicular lever F − Q</dt><dd>{vector(sasho.leverArmM, "m")}</dd></div>
        <div><dt className="text-slate-400">Complete angular velocity ω</dt><dd>{vector(metrics.angularVelocityRadS, "rad/s")}</dd></div>
        <div><dt className="text-slate-400">Rotation-only velocity</dt><dd>{vector(sasho.velocityMps, "m/s")}</dd></div>
        <div><dt className="text-slate-400">Vertical / horizontal speed</dt><dd>{number(sasho.velocityMps[1], "m/s", 4)} / {number(Math.hypot(sasho.velocityMps[0], sasho.velocityMps[2]), "m/s", 4)}</dd></div>
      </dl>
      <p className="mt-2 text-xs text-slate-400">
        <b className="text-slate-300">Geometry Basis:</b> {metrics.geometryBasis}.{" "}
        <b className="text-slate-300">Model Boundary:</b> {metrics.modelLimitations}
        {" "}<b className="text-slate-300">D-Plane State:</b> {dplane.status.split("_").join(" ")}. Geometry alone does not predict launch or ball spin without the declared collision model.
      </p>
    </aside>
  );
}
