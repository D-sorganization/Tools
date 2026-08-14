import { useEffect, useState } from "react";

import {
  defaultBallSetupForClub,
  type BallSetup,
} from "../model/ballSetup";
import { saveBallSetupPreference } from "../model/ballSetupPersistence";
import type { ClubSpec } from "../model/club";

export interface ControlledBallSetupProps {
  readonly ballSetup?: BallSetup;
  readonly ballSetupUserOverridden?: boolean;
  readonly ballSetupMessage?: string | null;
  readonly onBallSetupChange?: (setup: BallSetup) => void;
  readonly onBallSetupUserOverriddenChange?: (overridden: boolean) => void;
  readonly onBallSetupMessageChange?: (message: string | null) => void;
}

interface BallSetupController {
  readonly ballSetup: BallSetup;
  readonly ballSetupUserOverridden: boolean;
  readonly ballSetupMessage: string | null;
  readonly setOverride: (setup: BallSetup) => string | null;
  readonly restoreClubDefault: () => void;
  readonly setMessage: (message: string | null) => void;
}

/** Bridge app-owned ball state while preserving standalone panel compatibility. */
export function useSimulationBallSetup(
  club: ClubSpec | null,
  controlled: ControlledBallSetupProps,
): BallSetupController {
  const [internalSetup, setInternalSetup] = useState<BallSetup>(() =>
    defaultBallSetupForClub(club));
  const [internalOverridden, setInternalOverridden] = useState(false);
  const [internalMessage, setInternalMessage] = useState<string | null>(null);
  const ballSetup = controlled.ballSetup ?? internalSetup;
  const overridden = controlled.ballSetupUserOverridden ?? internalOverridden;
  const message = controlled.ballSetupMessage === undefined
    ? internalMessage
    : controlled.ballSetupMessage;
  const setSetup = controlled.onBallSetupChange ?? setInternalSetup;
  const setOverridden = controlled.onBallSetupUserOverriddenChange ??
    setInternalOverridden;
  const setMessage = controlled.onBallSetupMessageChange ?? setInternalMessage;

  const restoreClubDefault = () => {
    const next = defaultBallSetupForClub(club);
    setSetup(next);
    setOverridden(false);
    setMessage(saveBallSetupPreference({ setup: next, userOverridden: false }));
  };
  useEffect(() => {
    if (overridden) return;
    const next = defaultBallSetupForClub(club);
    setSetup(next);
    const warning = saveBallSetupPreference({ setup: next, userOverridden: false });
    if (warning !== null) setMessage(warning);
    // Stable React setters or parent callbacks own mutation; club/default
    // provenance is the only trigger for this reconciliation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [club, overridden]);

  return {
    ballSetup,
    ballSetupUserOverridden: overridden,
    ballSetupMessage: message,
    setOverride: (next) => {
      setSetup(next);
      setOverridden(true);
      const warning = saveBallSetupPreference({ setup: next, userOverridden: true });
      setMessage(warning);
      return warning;
    },
    restoreClubDefault,
    setMessage,
  };
}
