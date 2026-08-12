/** App-owned lifecycle for one exact imported regional-ground authority job. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  type RegionalGroundPreparationAuthorityClient,
} from "../model/regionalGroundAuthorityClient";
import type {
  ExecutionJobLaunch,
  RegionalGroundExecutionJob,
} from "../model/regionalGroundExecutionJob";
import {
  readRegionalGroundExecutionJobFile,
  type RegionalGroundExecutionJobFile,
} from "../model/regionalGroundExecutionJobFiles";
import {
  parseRegionalGroundJobPreparationRequest,
  REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
} from "../model/regionalGroundJobPreparationRequest";
import {
  stableRegionalGroundVariationRequestJson,
} from "../model/regionalGroundVariationRequestWire";
import { canonicalGroundJson } from "../model/flightGroundContract";
import type {
  RegionalGroundVariationRequestPort,
} from "../model/regionalGroundVariationWorkspace";

export interface RegionalGroundCurrentEditorSource {
  readonly launch: ExecutionJobLaunch;
  readonly variationRequestPort: RegionalGroundVariationRequestPort;
}

export interface RegionalGroundExecutionWorkspaceOptions {
  readonly client?: RegionalGroundAuthorityClient;
  readonly authority?: RegionalGroundAuthorityOptions;
  readonly executionPollIntervalMs?: number;
  readonly preparationSource?: RegionalGroundCurrentEditorSource;
  readonly preparationJobIdFactory?: () => string;
}

export interface RegionalGroundExecutionWorkspace {
  readonly authority: ReturnType<typeof useRegionalGroundAuthority>;
  readonly execution: RegionalGroundExecutionController;
  readonly acceptedJob: RegionalGroundExecutionJob | null;
  readonly sourceName: string | null;
  readonly confirmed: boolean;
  readonly importFile: (file: RegionalGroundExecutionJobFile) => Promise<void>;
  readonly preparationAvailable: boolean;
  readonly preparedJobStale: boolean;
  readonly prepareCurrentJob: () => Promise<"accepted" | "stale">;
  readonly setConfirmed: (confirmed: boolean) => void;
  readonly clear: () => void;
  readonly run: () => Promise<void>;
}

let preparationSequence = 0;
const defaultPreparationJobId = (): string => {
  preparationSequence += 1;
  return `editor-ground-${Date.now()}-${preparationSequence}`;
};

interface PreparationInputs {
  readonly launch: ExecutionJobLaunch;
  readonly variationRequest: Readonly<Record<string, unknown>>;
  readonly fingerprint: string;
}

/** Canonical identity of only the fields sent to the preparation authority. */
const preparationInputs = (
  source: RegionalGroundCurrentEditorSource,
): PreparationInputs => {
  const variationRequest = JSON.parse(
    stableRegionalGroundVariationRequestJson(source.variationRequestPort.snapshot()),
  ) as Readonly<Record<string, unknown>>;
  const validated = parseRegionalGroundJobPreparationRequest({
    schema_version: REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
    unit_system: "SI",
    job_id: "editor-ground-input-fingerprint",
    launch: source.launch,
    variation_request: variationRequest,
  });
  return {
    launch: validated.launch,
    variationRequest: validated.variation_request,
    fingerprint: canonicalGroundJson({
      launch: validated.launch,
      variation_request: validated.variation_request,
    }),
  };
};

const availablePreparationFingerprint = (
  source: RegionalGroundCurrentEditorSource | undefined,
): string | null => {
  if (source === undefined) return null;
  try {
    return preparationInputs(source).fingerprint;
  } catch {
    return null;
  }
};

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
  const preparationSource = useRef(options.preparationSource);
  preparationSource.current = options.preparationSource;
  const executionActive = useRef(execution.controls.statusEnabled);
  executionActive.current = execution.controls.statusEnabled;
  const [acceptedJob, setAcceptedJob] = useState<RegionalGroundExecutionJob | null>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [confirmed, setConfirmedState] = useState(false);
  const [acceptedPreparationFingerprint, setAcceptedPreparationFingerprint] =
    useState<string | null>(null);
  const [preparedJobPermanentlyStale, setPreparedJobPermanentlyStale] = useState(false);
  const preparationClient = client as Partial<RegionalGroundPreparationAuthorityClient>;
  const currentPreparationFingerprint = useMemo(
    () => availablePreparationFingerprint(options.preparationSource),
    [options.preparationSource],
  );
  const preparationAvailable = currentPreparationFingerprint !== null &&
    typeof preparationClient.prepare === "function";
  const preparedJobInputsDiffer = acceptedPreparationFingerprint !== null &&
    acceptedPreparationFingerprint !== currentPreparationFingerprint;
  const preparedJobStale = preparedJobPermanentlyStale || preparedJobInputsDiffer;

  useEffect(() => {
    if (!preparedJobInputsDiffer) return;
    setPreparedJobPermanentlyStale(true);
    setConfirmedState(false);
  }, [preparedJobInputsDiffer]);

  const acceptJob = useCallback((candidate: RegionalGroundExecutionJob, name: string,
    preparationFingerprint: string | null = null): void => {
    if (executionActive.current) {
      throw new Error("cannot replace an active or uncertain regional-ground execution job");
    }
    if (execution.job !== null) execution.reset();
    setAcceptedJob(candidate);
    setSourceName(name);
    setAcceptedPreparationFingerprint(preparationFingerprint);
    setPreparedJobPermanentlyStale(false);
    setConfirmedState(false);
  }, [execution]);

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
    acceptJob(candidate, file.name);
  }, [acceptJob]);

  const prepareCurrentJob = useCallback(async (): Promise<"accepted" | "stale"> => {
    if (executionActive.current) {
      throw new Error("cannot replace an active or uncertain regional-ground execution job");
    }
    const source = preparationSource.current;
    const prepare = preparationClient.prepare;
    if (source === undefined || typeof prepare !== "function") {
      throw new Error("current-editor job preparation is unavailable");
    }
    const candidateGeneration = generation.current + 1;
    generation.current = candidateGeneration;
    const inputs = preparationInputs(source);
    const request = parseRegionalGroundJobPreparationRequest({
      schema_version: REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
      unit_system: "SI",
      job_id: (options.preparationJobIdFactory ?? defaultPreparationJobId)(),
      launch: inputs.launch,
      variation_request: inputs.variationRequest,
    });
    const candidate = await prepare(request);
    if (candidateGeneration !== generation.current ||
        inputs.fingerprint !== availablePreparationFingerprint(preparationSource.current)) {
      return "stale";
    }
    acceptJob(candidate, "Current editors (Python prepared)", inputs.fingerprint);
    return "accepted";
  }, [acceptJob, options.preparationJobIdFactory, preparationClient.prepare]);

  const setConfirmed = useCallback((value: boolean): void => {
    if (execution.controls.statusEnabled || preparedJobStale) return;
    setConfirmedState(value);
  }, [execution.controls.statusEnabled, preparedJobStale]);

  const clear = useCallback((): void => {
    if (execution.controls.statusEnabled) {
      throw new Error("cannot clear an active or uncertain regional-ground execution job");
    }
    generation.current += 1;
    if (execution.job !== null) execution.reset();
    setAcceptedJob(null);
    setSourceName(null);
    setAcceptedPreparationFingerprint(null);
    setPreparedJobPermanentlyStale(false);
    setConfirmedState(false);
  }, [execution]);

  const run = useCallback(async (): Promise<void> => {
    if (acceptedJob === null || !confirmed || preparedJobStale) {
      throw new Error("an exact current accepted job and explicit confirmation are required");
    }
    await execution.submit(acceptedJob);
  }, [acceptedJob, confirmed, execution, preparedJobStale]);

  return { authority, execution, acceptedJob, sourceName, confirmed,
    importFile, preparationAvailable, preparedJobStale, prepareCurrentJob,
    setConfirmed, clear, run };
}
