import {
  JointLockConfig,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  withJointLocks,
  type DoublePendulumRunConfig,
  type PendulumState,
} from "../model/doublePendulum";
import { DecimalInput } from "./DecimalInput";

const DEGREES_PER_RADIAN = 180 / Math.PI;
const NUMBER_INPUT =
  "mt-1 w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 disabled:cursor-not-allowed disabled:opacity-40";

interface Props {
  initialState: PendulumState;
  runConfig: DoublePendulumRunConfig;
  onInitialStateChange: (state: PendulumState) => void;
  onRunConfigChange: (config: DoublePendulumRunConfig) => void;
}

interface JointControlProps {
  jointName: "Shoulder" | "Wrist";
  coordinateDescription: string;
  lockDescription: string;
  angleRad: number;
  velocityRadS: number;
  locked: boolean;
  onAngleChange: (valueRad: number) => void;
  onVelocityChange: (valueRadS: number) => void;
  onLockChange: (locked: boolean) => void;
}

function JointControl({
  jointName,
  coordinateDescription,
  lockDescription,
  angleRad,
  velocityRadS,
  locked,
  onAngleChange,
  onVelocityChange,
  onLockChange,
}: JointControlProps) {
  return (
    <fieldset className="rounded-lg border border-slate-700/80 bg-slate-950/40 p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-300">
        {jointName} Joint
      </legend>
      <p className="mb-2 text-xs leading-relaxed text-slate-500">
        {coordinateDescription}
      </p>
      <div className="grid grid-cols-2 gap-2">
        <label className="text-xs text-slate-400">
          Initial Angle (deg)
          <DecimalInput
            aria-label={`${jointName} Initial Angle`}
            value={angleRad * DEGREES_PER_RADIAN}
            onCommit={(value) => onAngleChange(value / DEGREES_PER_RADIAN)}
            min={-360}
            max={360}
            title={`${jointName} generalized coordinate at simulation time zero. ${coordinateDescription}`}
            className={NUMBER_INPUT}
          />
        </label>
        <label className="text-xs text-slate-400">
          Initial Velocity (deg/s)
          <DecimalInput
            aria-label={`${jointName} Initial Angular Velocity`}
            value={velocityRadS * DEGREES_PER_RADIAN}
            onCommit={(value) => onVelocityChange(value / DEGREES_PER_RADIAN)}
            min={-5000}
            max={5000}
            disabled={locked}
            title={locked
              ? `${jointName} initial angular velocity is fixed at zero while the ideal joint lock is active.`
              : `${jointName} generalized-coordinate angular velocity at simulation time zero.`}
            className={NUMBER_INPUT}
          />
        </label>
      </div>
      <label className="mt-2 flex cursor-pointer items-start gap-2 rounded border border-slate-700/70 px-2 py-2 text-xs text-slate-300 hover:border-sky-500/60">
        <input
          type="checkbox"
          aria-label={`Lock ${jointName} Joint`}
          checked={locked}
          onChange={(event) => onLockChange(event.target.checked)}
          title={lockDescription}
          className="mt-0.5 h-4 w-4 accent-sky-500"
        />
        <span>
          <span className="block font-semibold">Lock {jointName}</span>
          <span className="block text-slate-500">{lockDescription}</span>
        </span>
      </label>
    </fieldset>
  );
}

export function JointLockControls({
  initialState,
  runConfig,
  onInitialStateChange,
  onRunConfigChange,
}: Props) {
  const shoulderLocked = runConfig.jointLocks.isLocked(SHOULDER_JOINT_ID);
  const wristLocked = runConfig.jointLocks.isLocked(WRIST_JOINT_ID);

  const updateCoordinate = (index: number, value: number) => {
    const next = [...initialState] as PendulumState;
    next[index] = value;
    onInitialStateChange(next);
  };

  const updateLock = (jointId: string, locked: boolean) => {
    const ids = runConfig.jointLocks.lockedJointIds.filter((id) => id !== jointId);
    if (locked) ids.push(jointId);
    const velocityIndex = jointId === SHOULDER_JOINT_ID ? 2 : 3;
    if (locked) updateCoordinate(velocityIndex, 0);
    onRunConfigChange(withJointLocks(runConfig, new JointLockConfig(ids)));
  };

  return (
    <section aria-label="Double-pendulum joint constraints" className="mb-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Initial Joint State & Constraints
      </h3>
      <div className="grid gap-2">
        <JointControl
          jointName="Shoulder"
          coordinateDescription="θ₁ is an absolute angle measured relative to the fixed ground frame."
          lockDescription="Ideal absolute lock: holds shoulder angle θ₁ fixed relative to ground; constraint reaction is separate from commanded torque."
          angleRad={initialState[0]}
          velocityRadS={initialState[2]}
          locked={shoulderLocked}
          onAngleChange={(value) => updateCoordinate(0, value)}
          onVelocityChange={(value) => updateCoordinate(2, value)}
          onLockChange={(locked) => updateLock(SHOULDER_JOINT_ID, locked)}
        />
        <JointControl
          jointName="Wrist"
          coordinateDescription="θ₂ is a relative angle measured from the upper segment to the club segment."
          lockDescription="Ideal relative lock: holds wrist angle θ₂ fixed relative to the upper segment; constraint reaction is separate from commanded torque."
          angleRad={initialState[1]}
          velocityRadS={initialState[3]}
          locked={wristLocked}
          onAngleChange={(value) => updateCoordinate(1, value)}
          onVelocityChange={(value) => updateCoordinate(3, value)}
          onLockChange={(locked) => updateLock(WRIST_JOINT_ID, locked)}
        />
      </div>
      <p
        role="status"
        aria-label="Joint lock status"
        className="mt-2 rounded border border-sky-500/25 bg-slate-950/60 p-2 text-xs text-sky-200"
      >
        Shoulder (absolute): {shoulderLocked ? "locked" : "free"}; Wrist
        (relative): {wristLocked ? "locked" : "free"}.
      </p>
    </section>
  );
}
