/** Strict workspace selection for the canonical torque-profile library. */

import {
  passiveDoublePendulumRun,
  prescribedDoublePendulumRun,
  type DoublePendulumRunConfig,
} from "./doublePendulum";
import { JointLockConfig } from "./jointLocks";
import { PrescribedTorqueProfile } from "./torqueProfiles";

export const TORQUE_WORKSPACE_SCHEMA =
  "rate_of_closure.torque_workspace_selection";
export const TORQUE_WORKSPACE_SCHEMA_VERSION = 1;

export interface TorqueWorkspaceSnapshot {
  readonly profiles: readonly PrescribedTorqueProfile[];
  readonly activeProfileId: string | null;
  readonly runConfig: DoublePendulumRunConfig;
}

export interface TorqueSessionMigrationInput {
  readonly isLegacy: boolean;
  readonly selectionDocument: unknown;
  readonly profileDocuments: unknown;
  readonly legacyFallback?: TorqueWorkspaceSnapshot;
}

function record(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactRecord(
  value: unknown,
  fields: readonly string[],
  context: string,
): Record<string, unknown> {
  const data = record(value, context);
  const actual = Object.keys(data);
  if (actual.length !== fields.length || fields.some((field) => !(field in data))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return data;
}

function activeProfile(
  profiles: readonly PrescribedTorqueProfile[],
  activeProfileId: string | null,
): PrescribedTorqueProfile | null {
  if (activeProfileId === null) return null;
  return profiles.find((profile) => profile.profileId === activeProfileId) ?? null;
}

/** Validate library identity and return deterministically ordered state. */
export function validatedTorqueWorkspace(
  snapshot: TorqueWorkspaceSnapshot,
): TorqueWorkspaceSnapshot {
  if (!Array.isArray(snapshot?.profiles)) {
    throw new TypeError("profiles must be an array");
  }
  const profiles = snapshot.profiles.map((profile) => {
    if (!(profile instanceof PrescribedTorqueProfile)) {
      throw new TypeError("profiles must contain PrescribedTorqueProfile values");
    }
    return profile;
  });
  if (new Set(profiles.map((profile) => profile.profileId)).size !== profiles.length) {
    throw new RangeError("torque profile IDs must be unique");
  }
  if (snapshot.activeProfileId !== null &&
      typeof snapshot.activeProfileId !== "string") {
    throw new TypeError("activeProfileId must be a stable string or null");
  }
  if (snapshot.runConfig?.mode !== "passive" &&
      snapshot.runConfig?.mode !== "prescribed") {
    throw new RangeError("unsupported torque run mode");
  }
  if (!(snapshot.runConfig.jointLocks instanceof JointLockConfig)) {
    throw new TypeError("runConfig jointLocks must be a JointLockConfig");
  }
  if (profiles.length > 0 && snapshot.activeProfileId === null) {
    throw new RangeError("a non-empty torque library requires an active profile");
  }
  const selected = activeProfile(profiles, snapshot.activeProfileId);
  if (snapshot.activeProfileId !== null && selected === null) {
    throw new RangeError("active torque profile is not present in the library");
  }
  if (snapshot.runConfig.mode === "prescribed" &&
      snapshot.runConfig.profile.profileId !== snapshot.activeProfileId) {
    throw new RangeError("prescribed run profile must be the active profile");
  }
  const ordered = [...profiles].sort((left, right) =>
    left.profileId.localeCompare(right.profileId));
  const canonicalSelected = activeProfile(ordered, snapshot.activeProfileId);
  const runConfig = snapshot.runConfig.mode === "prescribed"
    ? prescribedDoublePendulumRun(canonicalSelected!, snapshot.runConfig.jointLocks)
    : passiveDoublePendulumRun(snapshot.runConfig.jointLocks);
  return Object.freeze({
    profiles: Object.freeze(ordered),
    activeProfileId: snapshot.activeProfileId,
    runConfig,
  });
}

function provenance(snapshot: TorqueWorkspaceSnapshot) {
  const profile = activeProfile(snapshot.profiles, snapshot.activeProfileId);
  return {
    kind: profile === null ? "none" : "library_profile",
    profile_source: profile?.source ?? null,
  };
}

/** Serialize selection only; canonical profiles remain at the workspace root. */
export function torqueWorkspaceDocument(
  snapshot: TorqueWorkspaceSnapshot,
): Record<string, unknown> {
  const state = validatedTorqueWorkspace(snapshot);
  const prescribedId = state.runConfig.mode === "prescribed"
    ? state.runConfig.profile.profileId
    : null;
  return {
    schema: TORQUE_WORKSPACE_SCHEMA,
    schema_version: TORQUE_WORKSPACE_SCHEMA_VERSION,
    data: {
      active_profile_id: state.activeProfileId,
      run_config: {
        mode: state.runConfig.mode,
        prescribed_profile_id: prescribedId,
        locked_joint_ids: [...state.runConfig.jointLocks.lockedJointIds],
      },
      selection_provenance: provenance(state),
    },
  };
}

function runConfig(
  value: unknown,
  profiles: readonly PrescribedTorqueProfile[],
): DoublePendulumRunConfig {
  const data = exactRecord(
    value,
    ["mode", "prescribed_profile_id", "locked_joint_ids"],
    "torque run_config",
  );
  if (!Array.isArray(data.locked_joint_ids)) {
    throw new TypeError("locked_joint_ids must be a JSON array");
  }
  const locks = new JointLockConfig(data.locked_joint_ids as string[]);
  if (data.mode === "passive") {
    if (data.prescribed_profile_id !== null) {
      throw new RangeError("passive run mode cannot prescribe a profile");
    }
    return passiveDoublePendulumRun(locks);
  }
  if (data.mode !== "prescribed" || typeof data.prescribed_profile_id !== "string") {
    throw new RangeError("unsupported torque run mode or profile selection");
  }
  const profile = activeProfile(profiles, data.prescribed_profile_id);
  if (profile === null) {
    throw new RangeError("prescribed torque profile is not present in the library");
  }
  return prescribedDoublePendulumRun(profile, locks);
}

/** Parse selection and validate its provenance against the root library. */
export function torqueWorkspaceFromDocument(
  value: unknown,
  profiles: readonly PrescribedTorqueProfile[],
): TorqueWorkspaceSnapshot {
  const envelope = exactRecord(
    value,
    ["schema", "schema_version", "data"],
    "torque workspace",
  );
  if (envelope.schema !== TORQUE_WORKSPACE_SCHEMA ||
      envelope.schema_version !== TORQUE_WORKSPACE_SCHEMA_VERSION) {
    throw new RangeError("unsupported torque workspace selection payload");
  }
  const data = exactRecord(
    envelope.data,
    ["active_profile_id", "run_config", "selection_provenance"],
    "torque workspace.data",
  );
  const activeId = data.active_profile_id;
  if (activeId !== null && typeof activeId !== "string") {
    throw new TypeError("active_profile_id must be a stable string or null");
  }
  const parsedRun = runConfig(data.run_config, profiles);
  const selected = activeProfile(profiles, activeId);
  const state = validatedTorqueWorkspace({
    profiles,
    activeProfileId: activeId,
    runConfig: parsedRun,
  });
  const declared = exactRecord(
    data.selection_provenance,
    ["kind", "profile_source"],
    "torque selection_provenance",
  );
  const expected = provenance(state);
  if (declared.kind !== expected.kind ||
      declared.profile_source !== expected.profile_source ||
      (selected === null && activeId !== null)) {
    throw new RangeError("torque selection provenance does not match profile source");
  }
  return state;
}

/** Preserve explicit live state unless a legacy root library conflicts. */
export function migratedLegacyTorqueFallback(
  fallback: TorqueWorkspaceSnapshot,
  documentProfiles: readonly PrescribedTorqueProfile[],
): TorqueWorkspaceSnapshot {
  const state = validatedTorqueWorkspace(fallback);
  if (documentProfiles.length > 0) {
    const current = JSON.stringify(state.profiles.map((profile) => profile.toJsonObject()));
    const saved = JSON.stringify(
      [...documentProfiles]
        .sort((left, right) => left.profileId.localeCompare(right.profileId))
        .map((profile) => profile.toJsonObject()),
    );
    if (current !== saved) {
      throw new RangeError(
        "legacy workspace torque library conflicts with the explicit fallback",
      );
    }
  }
  return state;
}

/** Resolve current or explicit-legacy torque state from workspace components. */
export function torqueWorkspaceFromSession(
  input: TorqueSessionMigrationInput,
): TorqueWorkspaceSnapshot {
  if (!Array.isArray(input.profileDocuments)) {
    throw new TypeError("prescribed_torque_profiles must be a JSON array");
  }
  const profiles = input.profileDocuments.map(
    PrescribedTorqueProfile.fromJsonObject,
  );
  if (!input.isLegacy) {
    return torqueWorkspaceFromDocument(input.selectionDocument, profiles);
  }
  if (input.legacyFallback === undefined) {
    throw new RangeError(
      "legacy model_session requires an explicit torque migration fallback",
    );
  }
  return migratedLegacyTorqueFallback(input.legacyFallback, profiles);
}
