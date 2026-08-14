/** Public web-session records, separated from the simulation orchestrator. */

import { type ClubAssemblyBinding } from "./clubAssemblyBinding";
import { type ClubSpec } from "./club";
import { type ClubAssemblyImpactInputs } from "./clubAssemblySimulationAdapter";
import type { BallSetup } from "./ballSetup";
import type { ContactMode, ImpactOutcomeTs } from "./contact";
import type {
  DoublePendulumRunConfig,
  PendulumParams,
  PendulumState,
  summarizeDoublePendulumRun,
} from "./doublePendulum";
import type { AngularFlightPoint } from "./flight";
import type { ImpactClubProperties, Vec3 } from "./impactPhysics";
import type { ManualDelivery, ShaftAxisDatum } from "./manualDelivery";
import type { Mat3 } from "./rotation";

export type WebSourceKind = "manual" | "double_pendulum" | "triple_pendulum";

export interface SimulationInput {
  sourceKind: WebSourceKind;
  clubheadSpeedMph: number;
  omegaDps: Vec3;
  loftDeg: number;
  impactOffsetToeMm: number;
  impactOffsetHighMm: number;
  planeYawDeg: number;
  planeSideTiltDeg: number;
  planeForwardTiltDeg: number;
  impactTimeS: number | null;
  swingDurationS: number;
  pendulumParameters?: PendulumParams;
  impactTimeOffsetS?: number;
  club?: ImpactClubProperties;
  contactMode?: ContactMode;
  doublePendulumRun?: DoublePendulumRunConfig;
  doublePendulumInitialState?: PendulumState;
  ballSetup?: BallSetup;
  manualAttackAngleDeg?: number;
  manualClubPathDeg?: number;
  manualForwardShaftLeanDeg?: number;
  shaftAxisDatum?: ShaftAxisDatum;
  /** Qualified binding for the exact selected club (#4111 / #4341). */
  assemblyBinding?: ClubAssemblyBinding;
  /** Full selected specification required to validate an assembly binding. */
  assemblyClubSpec?: ClubSpec;
}

export interface SwingSampleTs {
  t: number;
  position: Vec3;
  velocity: Vec3;
  angularVelocity: Vec3;
  rotation: Mat3;
  joints: Vec3[];
}

export interface SimulationLaunchTs {
  ballSpeedMph: number;
  launchAngleDeg: number;
  launchAzimuthDeg: number;
  spinRpm: number;
  carryM: number;
  maxHeightM: number;
  flightTimeS: number;
  landingAngleDeg: number;
}

export interface SimulationRunTs {
  sourceKind: WebSourceKind;
  torqueRun: ReturnType<typeof summarizeDoublePendulumRun>;
  swing: SwingSampleTs[];
  impactOutcome: ImpactOutcomeTs;
  impactTimeS: number | null;
  totalDurationS: number;
  launch: SimulationLaunchTs | null;
  flight: AngularFlightPoint[];
  ballSetup: BallSetup;
  ballPositionM: Vec3;
  manualDelivery: ManualDelivery;
  clubAssemblyUsage: ClubAssemblyImpactInputs;
}
