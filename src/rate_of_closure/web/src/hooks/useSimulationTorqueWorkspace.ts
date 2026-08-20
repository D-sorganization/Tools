import { useState, type Dispatch, type SetStateAction } from "react";

import {
  passiveDoublePendulumRun,
  prescribedDoublePendulumRun,
  type DoublePendulumRunConfig,
} from "../model/doublePendulum";
import {
  loadTorqueProfileLibrary,
  starterTorqueProfile,
} from "../model/torqueProfileEditor";
import type { PrescribedTorqueProfile } from "../model/torqueProfiles";
import {
  validatedTorqueWorkspace,
  type TorqueWorkspaceSnapshot,
} from "../model/workspaceTorqueSession";

export interface ControlledTorqueWorkspaceProps {
  readonly torqueWorkspace?: TorqueWorkspaceSnapshot;
  readonly onTorqueWorkspaceChange?: Dispatch<
    SetStateAction<TorqueWorkspaceSnapshot>
  >;
}

export function loadInitialTorqueWorkspace(): TorqueWorkspaceSnapshot {
  let profiles: readonly PrescribedTorqueProfile[];
  try {
    profiles = loadTorqueProfileLibrary();
  } catch {
    profiles = Object.freeze([starterTorqueProfile()]);
  }
  return validatedTorqueWorkspace({
    profiles,
    activeProfileId: profiles[0].profileId,
    runConfig: passiveDoublePendulumRun(),
  });
}

/** Own torque state locally or bridge the app-level workspace authority. */
export function useSimulationTorqueWorkspace(
  controlled: ControlledTorqueWorkspaceProps,
) {
  const [internal, setInternal] = useState(loadInitialTorqueWorkspace);
  const state = controlled.torqueWorkspace ?? internal;
  const setState = controlled.onTorqueWorkspaceChange ?? setInternal;
  const setRunConfig = (runConfig: DoublePendulumRunConfig) => {
    setState((current) => validatedTorqueWorkspace({
      ...current,
      activeProfileId: runConfig.mode === "prescribed"
        ? runConfig.profile.profileId
        : current.activeProfileId,
      runConfig,
    }));
  };
  const setLibrary = (
    profiles: readonly PrescribedTorqueProfile[],
    activeProfileId: string,
  ) => {
    setState((current) => {
      const selected = profiles.find((profile) =>
        profile.profileId === activeProfileId);
      if (selected === undefined) {
        throw new RangeError("active torque profile is not present in the library");
      }
      const runConfig = current.runConfig.mode === "prescribed"
        ? prescribedDoublePendulumRun(selected, current.runConfig.jointLocks)
        : passiveDoublePendulumRun(current.runConfig.jointLocks);
      return validatedTorqueWorkspace({ profiles, activeProfileId, runConfig });
    });
  };
  return { state, setRunConfig, setLibrary };
}
