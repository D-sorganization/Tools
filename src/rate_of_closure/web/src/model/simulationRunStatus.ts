/** Build one consistent accessible run-state summary for the workspace. */

import { SHOULDER_JOINT_ID, WRIST_JOINT_ID } from "./doublePendulum";
import { type SimulationRunTs } from "./simulation";

export function simulationRunStatus(
  run: SimulationRunTs | null,
  runError: string | null,
  runIsStale: boolean,
): string {
  if (runError) return `Run failed: ${runError}`;
  if (runIsStale) return "Inputs changed — run required";
  if (!run) return "Not run";
  const completed =
    run.impactOutcome.status === "miss"
      ? "Completed — no club–ball impact"
      : "Completed — impact and flight available";
  const details = [
    run.torqueRun.mode === "prescribed"
      ? `prescribed torque profile ${run.torqueRun.profileId}`
      : null,
    run.torqueRun.lockedJointIds.includes(SHOULDER_JOINT_ID)
      ? "Shoulder locked (absolute ground frame)"
      : null,
    run.torqueRun.lockedJointIds.includes(WRIST_JOINT_ID)
      ? "Wrist locked (relative upper-segment frame)"
      : null,
  ].filter((detail): detail is string => detail !== null);
  return [completed, ...details].join("; ");
}
