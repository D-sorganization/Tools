import { useMemo, useState, type Dispatch, type SetStateAction } from "react";

import type { UnitSelections } from "../components/ImpactExplorerPanel";
import { getClub, type ClubSpec } from "../model/club";
import { generatedHeadFor, type GeneratedHead } from "../model/clubHeadGeneration";
import { DEFAULT_SCENARIO, type ImpactScenario } from "../model/impact";
import {
  defaultBallSetupForClub,
  type BallSetup,
} from "../model/ballSetup";
import { loadBallSetupPreference } from "../model/ballSetupPersistence";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";

export interface ImpactAppModel {
  readonly scenario: ImpactScenario;
  readonly setScenario: Dispatch<SetStateAction<ImpactScenario>>;
  readonly spatialTarget: SpatialTargetTs;
  readonly setSpatialTarget: Dispatch<SetStateAction<SpatialTargetTs>>;
  readonly ballSetup: BallSetup;
  readonly setBallSetup: Dispatch<SetStateAction<BallSetup>>;
  readonly ballSetupUserOverridden: boolean;
  readonly setBallSetupUserOverridden: Dispatch<SetStateAction<boolean>>;
  readonly ballSetupMessage: string | null;
  readonly setBallSetupMessage: Dispatch<SetStateAction<string | null>>;
  readonly units: UnitSelections;
  readonly setUnits: Dispatch<SetStateAction<UnitSelections>>;
  readonly generatedHead: GeneratedHead;
  readonly setGeneratedHead: Dispatch<SetStateAction<GeneratedHead>>;
  readonly clubSpec: ClubSpec;
  readonly setClubSpec: Dispatch<SetStateAction<ClubSpec>>;
  readonly explained: string;
  readonly setExplained: Dispatch<SetStateAction<string>>;
  readonly glossaryTerm: string | undefined;
  readonly setGlossaryTerm: Dispatch<SetStateAction<string | undefined>>;
}

const DEFAULT_UNITS: UnitSelections = {
  speed: "mph",
  rotation: "deg/s",
  length: "mm",
  distance: "yd",
};

export function useImpactAppModel(): ImpactAppModel {
  const defaultDriver = useMemo(() => getClub("Driver 10.5°"), []);
  const [initialBallPreference] = useState(() => {
    const clubDefault = defaultBallSetupForClub(defaultDriver);
    const loaded = loadBallSetupPreference(undefined, clubDefault);
    return !loaded.userOverridden && loaded.warning === null
      ? { ...loaded, setup: clubDefault }
      : loaded;
  });
  const [scenario, setScenario] = useState(DEFAULT_SCENARIO);
  const [spatialTarget, setSpatialTarget] = useState(() =>
    spatialTargetFromRegion(DEFAULT_TARGET));
  const [ballSetup, setBallSetup] = useState(initialBallPreference.setup);
  const [ballSetupUserOverridden, setBallSetupUserOverridden] = useState(
    initialBallPreference.userOverridden,
  );
  const [ballSetupMessage, setBallSetupMessage] = useState(
    initialBallPreference.warning,
  );
  const [units, setUnits] = useState(DEFAULT_UNITS);
  const [generatedHead, setGeneratedHead] = useState(() =>
    generatedHeadFor(defaultDriver));
  const [clubSpec, setClubSpec] = useState(defaultDriver);
  const [explained, setExplained] = useState("pathDeviationDeg");
  const [glossaryTerm, setGlossaryTerm] = useState<string>();
  return {
    scenario, setScenario, spatialTarget, setSpatialTarget,
    ballSetup, setBallSetup, ballSetupUserOverridden,
    setBallSetupUserOverridden, ballSetupMessage, setBallSetupMessage,
    units, setUnits,
    generatedHead, setGeneratedHead, clubSpec, setClubSpec, explained,
    setExplained, glossaryTerm, setGlossaryTerm,
  };
}
