import { useMemo, useState, type Dispatch, type SetStateAction } from "react";

import type { UnitSelections } from "../components/ImpactExplorerPanel";
import { getClub, type ClubSpec } from "../model/club";
import { generatedHeadFor, type GeneratedHead } from "../model/clubHeadGeneration";
import { DEFAULT_SCENARIO, type ImpactScenario } from "../model/impact";
import {
  DEFAULT_FLIGHT_EXPLORER_DRAFT,
  executionLaunchForFlightExplorerDraft,
  type FlightExplorerDraft,
} from "../model/flightPreparationLaunch";
import type { ExecutionJobLaunch } from "../model/regionalGroundExecutionJob";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";

export interface ImpactAppModel {
  readonly scenario: ImpactScenario;
  readonly setScenario: Dispatch<SetStateAction<ImpactScenario>>;
  readonly spatialTarget: SpatialTargetTs;
  readonly setSpatialTarget: Dispatch<SetStateAction<SpatialTargetTs>>;
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
  readonly flightExplorerDraft: FlightExplorerDraft;
  readonly setFlightExplorerDraft: Dispatch<SetStateAction<FlightExplorerDraft>>;
  readonly flightPreparationLaunch: ExecutionJobLaunch | null;
}

const DEFAULT_UNITS: UnitSelections = {
  speed: "mph",
  rotation: "deg/s",
  length: "mm",
  distance: "yd",
};

export function useImpactAppModel(): ImpactAppModel {
  const defaultDriver = useMemo(() => getClub("Driver 10.5°"), []);
  const [scenario, setScenario] = useState(DEFAULT_SCENARIO);
  const [spatialTarget, setSpatialTarget] = useState(() =>
    spatialTargetFromRegion(DEFAULT_TARGET));
  const [units, setUnits] = useState(DEFAULT_UNITS);
  const [generatedHead, setGeneratedHead] = useState(() =>
    generatedHeadFor(defaultDriver));
  const [clubSpec, setClubSpec] = useState(defaultDriver);
  const [explained, setExplained] = useState("pathDeviationDeg");
  const [glossaryTerm, setGlossaryTerm] = useState<string>();
  const [flightExplorerDraft, setFlightExplorerDraft] = useState(
    DEFAULT_FLIGHT_EXPLORER_DRAFT,
  );
  const flightPreparationLaunch = useMemo(
    () => {
      try {
        return executionLaunchForFlightExplorerDraft(flightExplorerDraft);
      } catch {
        return null;
      }
    },
    [flightExplorerDraft],
  );
  return {
    scenario, setScenario, spatialTarget, setSpatialTarget, units, setUnits,
    generatedHead, setGeneratedHead, clubSpec, setClubSpec, explained,
    setExplained, glossaryTerm, setGlossaryTerm, flightExplorerDraft,
    setFlightExplorerDraft, flightPreparationLaunch,
  };
}
