/** App-owned lifecycle for one exact imported regional-ground authority job. */

import { useCallback, useMemo, useRef, useState } from "react";

import {
  useRegionalGroundAuthority,
  type RegionalGroundAuthorityOptions,
} from "./useRegionalGroundAuthority";
import {
  useRegionalGroundExecutionController,
  type RegionalGroundExecutionController,
} from "./useRegionalGroundExecutionController";
import {
  createRegionalGroundAuthorityClient,
  type RegionalGroundAuthorityClient,
} from "../model/regionalGroundAuthorityClient";
import type { RegionalGroundExecutionJob } from "../model/regionalGroundExecutionJob";
import {
  readRegionalGroundExecutionJobFile,
  type RegionalGroundExecutionJobFile,
} from "../model/regionalGroundExecutionJobFiles";

export interface RegionalGroundExecutionWorkspaceOptions {
  readonly client?: RegionalGroundAuthorityClient;
  readonly authority?: RegionalGroundAuthorityOptions;
  readonly executionPollIntervalMs?: number;
}

export interface RegionalGroundExecutionWorkspace {
  readonly authority: ReturnType<typeof useRegionalGroundAuthority>;
  readonly execution: RegionalGroundExecutionController;
  readonly acceptedJob: RegionalGroundExecutionJob | null;
  readonly sourceName: string | null;
  readonly confirmed: boolean;
  readonly importFile: (file: RegionalGroundExecutionJobFile) => Promise<void>;
  readonly setConfirmed: (confirmed: boolean) => void;
  readonly clear: () => void;
  readonly run: () => Promise<void>;
}

export function useRegionalGroundExecutionWorkspace(
  options: RegionalGroundExecutionWorkspaceOptions = {},
): RegionalGroundExecutionWorkspace {
  const client = useMemo(
    () => options.client ?? createRegionalGroundAuthorityClient(),
    [options.client],
  );
  const authority = useRegionalGroundAuthority(options.authority);
  const execution = useRegionalGroundExecutionController({
    client,
    capability: authority.capability,
    ...(options.executionPollIntervalMs === undefined
      ? {} : { pollIntervalMs: options.executionPollIntervalMs }),
  });
  const generation = useRef(0);
  const executionActive = useRef(execution.controls.statusEnabled);
  executionActive.current = execution.controls.statusEnabled;
  const [acceptedJob, setAcceptedJob] = useState<RegionalGroundExecutionJob | null>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [confirmed, setConfirmedState] = useState(false);

  const importFile = useCallback(async (file: RegionalGroundExecutionJobFile): Promise<void> => {
    if (executionActive.current) {
      throw new Error("cannot replace an active or uncertain regional-ground execution job");
    }
    const candidateGeneration = generation.current + 1;
    generation.current = candidateGeneration;
    const candidate = await readRegionalGroundExecutionJobFile(file);
    if (candidateGeneration !== generation.current) return;
    if (executionActive.current) {
      throw new Error("cannot replace an active or uncertain regional-ground execution job");
    }
    if (execution.job !== null) execution.reset();
    setAcceptedJob(candidate);
    setSourceName(file.name);
    setConfirmedState(false);
  }, [execution]);

  const setConfirmed = useCallback((value: boolean): void => {
    if (execution.controls.statusEnabled) return;
    setConfirmedState(value);
  }, [execution.controls.statusEnabled]);

  const clear = useCallback((): void => {
    if (execution.controls.statusEnabled) {
      throw new Error("cannot clear an active or uncertain regional-ground execution job");
    }
    generation.current += 1;
    if (execution.job !== null) execution.reset();
    setAcceptedJob(null);
    setSourceName(null);
    setConfirmedState(false);
  }, [execution]);

  const run = useCallback(async (): Promise<void> => {
    if (acceptedJob === null || !confirmed) {
      throw new Error("an exact imported job and explicit confirmation are required");
    }
    await execution.submit(acceptedJob);
  }, [acceptedJob, confirmed, execution]);

  return { authority, execution, acceptedJob, sourceName, confirmed,
    importFile, setConfirmed, clear, run };
}
