/** UI-neutral orchestration over the strict regional-ground authority client. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  parseRegionalGroundAuthorityCapability,
  type RegionalGroundAuthorityCapability,
} from "../model/regionalGroundAuthority";
import {
  RegionalGroundAuthorityRequestError,
  type RegionalGroundAuthorityClient,
  type RegionalGroundAuthorityJobFailure,
  type RegionalGroundAuthorityJobStatus,
} from "../model/regionalGroundAuthorityClient";
import type { RegionalGroundExecutionResult } from "../model/regionalGroundExecutionResult";
import {
  parseRegionalGroundExecutionJob,
  type RegionalGroundExecutionJob,
} from "../model/regionalGroundExecutionJob";
import type { RegionalGroundExecutionControlState } from "./useRegionalGroundAuthority";

const MIN_POLL_INTERVAL_MS = 250;
const DEFAULT_POLL_INTERVAL_MS = 1_000;

export type RegionalGroundExecutionPhase =
  | "idle"
  | "submitting"
  | "queued"
  | "running"
  | "cancel_requested"
  | "cancelling"
  | "retrieving_result"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "not_found"
  | "request_failed";

export interface RegionalGroundExecutionProgress {
  readonly completed: number;
  readonly total: number;
}

export interface RegionalGroundExecutionControllerState {
  readonly phase: RegionalGroundExecutionPhase;
  readonly job: RegionalGroundExecutionJob | null;
  readonly status: RegionalGroundAuthorityJobStatus | null;
  readonly progress: RegionalGroundExecutionProgress | null;
  readonly failure: RegionalGroundAuthorityJobFailure | null;
  readonly result: RegionalGroundExecutionResult | null;
  readonly error: Error | null;
}

export interface RegionalGroundExecutionControllerOptions {
  readonly client: RegionalGroundAuthorityClient;
  readonly capability: unknown;
  readonly pollIntervalMs?: number;
}

export interface RegionalGroundExecutionController
  extends RegionalGroundExecutionControllerState {
  readonly controls: RegionalGroundExecutionControlState;
  readonly submit: (job: RegionalGroundExecutionJob) => Promise<void>;
  readonly cancel: () => Promise<void>;
  readonly reconcile: () => Promise<void>;
  readonly reset: () => void;
}

const initialState = (): RegionalGroundExecutionControllerState => ({
  phase: "idle",
  job: null,
  status: null,
  progress: null,
  failure: null,
  result: null,
  error: null,
});

const qualifiedForExecution = (
  capability: RegionalGroundAuthorityCapability,
): boolean => capability.available && capability.regional_ground_execution;

const validateInterval = (value: number): number => {
  if (!Number.isSafeInteger(value) || value < MIN_POLL_INTERVAL_MS) {
    throw new RangeError(`pollIntervalMs must be an integer >= ${MIN_POLL_INTERVAL_MS}`);
  }
  return value;
};

const asError = (reason: unknown): Error =>
  reason instanceof Error ? reason : new Error("regional-ground authority request failed");

const isAbort = (reason: unknown): boolean =>
  reason instanceof DOMException && reason.name === "AbortError";

/** Coordinate one exact authority job without adding UI or browser physics. */
export function useRegionalGroundExecutionController(
  options: RegionalGroundExecutionControllerOptions,
): RegionalGroundExecutionController {
  const interval = validateInterval(options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
  const [state, setState] = useState(initialState);
  const mounted = useRef(true);
  const generation = useRef(0);
  const operation = useRef(0);
  const active = useRef(false);
  const submitted = useRef(false);
  const timer = useRef<number>();
  const controller = useRef<AbortController>();
  const client = useRef(options.client);
  const capability = useRef(options.capability);
  const pollInterval = useRef(interval);
  const job = useRef<RegionalGroundExecutionJob | null>(null);
  const processStatus = useRef<(
    value: RegionalGroundAuthorityJobStatus,
    run: number,
  ) => Promise<void>>();
  const poll = useRef<(run: number) => Promise<void>>();

  client.current = options.client;
  capability.current = options.capability;
  pollInterval.current = interval;

  const current = useCallback((run: number, request?: number): boolean =>
    mounted.current && generation.current === run &&
    (request === undefined || operation.current === request), []);

  const publish = useCallback((
    run: number,
    changed: Partial<RegionalGroundExecutionControllerState>,
  ): void => {
    if (!current(run)) return;
    setState((prior) => ({ ...prior, ...changed }));
  }, [current]);

  const clearPending = useCallback((): void => {
    if (timer.current !== undefined) window.clearTimeout(timer.current);
    timer.current = undefined;
    operation.current += 1;
    controller.current?.abort();
    controller.current = undefined;
  }, []);

  const failRequest = useCallback((run: number, reason: unknown): void => {
    if (!current(run) || isAbort(reason)) return;
    publish(run, { phase: "request_failed", error: asError(reason) });
  }, [current, publish]);

  const publishNotFound = useCallback((run: number, reason: unknown): boolean => {
    if (!(reason instanceof RegionalGroundAuthorityRequestError) ||
        reason.code !== "job_not_found" || !current(run)) return false;
    active.current = false;
    submitted.current = false;
    publish(run, { phase: "not_found", status: null, progress: null, failure: null,
      result: null, error: reason });
    return true;
  }, [current, publish]);

  const beginRequest = useCallback((): {
    readonly id: number;
    readonly signal: AbortSignal;
  } => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    operation.current += 1;
    return { id: operation.current, signal: next.signal };
  }, []);

  const schedulePoll = useCallback((run: number): void => {
    if (!current(run)) return;
    timer.current = window.setTimeout(() => { void poll.current?.(run); }, pollInterval.current);
  }, [current]);

  processStatus.current = async (value, run) => {
    if (!current(run)) return;
    const progress = Object.freeze({ completed: value.completed, total: value.total });
    const common = { status: value, progress, failure: value.failure, error: null };
    if (value.status === "failed" || value.status === "cancelled") {
      active.current = false;
      publish(run, { ...common, phase: value.status, result: null });
      return;
    }
    if (value.status !== "succeeded") {
      publish(run, { ...common, phase: value.status, result: null });
      schedulePoll(run);
      return;
    }
    publish(run, { ...common, phase: "retrieving_result", result: null });
    const sourceJob = job.current;
    if (sourceJob === null) return;
    const request = beginRequest();
    try {
      const complete = await client.current.result(sourceJob, request.signal);
      if (!current(run, request.id)) return;
      active.current = false;
      publish(run, { phase: "succeeded", result: complete });
    } catch (reason) {
      if (current(run, request.id)) failRequest(run, reason);
    }
  };

  poll.current = async (run) => {
    const sourceJob = job.current;
    if (!current(run) || sourceJob === null) return;
    const request = beginRequest();
    try {
      const next = await client.current.status(sourceJob, request.signal);
      if (current(run, request.id)) await processStatus.current?.(next, run);
    } catch (reason) {
      if (current(run, request.id) && !publishNotFound(run, reason)) {
        failRequest(run, reason);
      }
    }
  };

  const submit = useCallback(async (source: RegionalGroundExecutionJob): Promise<void> => {
    if (active.current) throw new Error("a regional-ground authority job is already active");
    if (submitted.current) {
      throw new Error("the accepted regional-ground authority job has already been submitted");
    }
    const exactCapability = parseRegionalGroundAuthorityCapability(capability.current);
    if (!qualifiedForExecution(exactCapability)) {
      throw new Error("regional-ground execution capability is unavailable");
    }
    const exactJob = parseRegionalGroundExecutionJob(source);
    clearPending();
    const run = generation.current + 1;
    generation.current = run;
    job.current = exactJob;
    active.current = true;
    submitted.current = true;
    publish(run, { ...initialState(), phase: "submitting", job: exactJob });
    const request = beginRequest();
    try {
      const submitted = await client.current.submit(exactJob, request.signal);
      if (current(run, request.id)) await processStatus.current?.(submitted, run);
    } catch (reason) {
      if (current(run, request.id)) {
        failRequest(run, reason);
        throw reason;
      }
    }
  }, [beginRequest, clearPending, current, failRequest, publish]);

  const reconcile = useCallback(async (): Promise<void> => {
    const sourceJob = job.current;
    if (!active.current || sourceJob === null) {
      throw new Error("no uncertain or active regional-ground authority job exists");
    }
    const run = generation.current;
    clearPending();
    const request = beginRequest();
    try {
      const next = await client.current.status(sourceJob, request.signal);
      if (current(run, request.id)) await processStatus.current?.(next, run);
    } catch (reason) {
      if (current(run, request.id) && !publishNotFound(run, reason)) {
        failRequest(run, reason);
        throw reason;
      }
    }
  }, [beginRequest, clearPending, current, failRequest, publishNotFound]);

  const cancel = useCallback(async (): Promise<void> => {
    const sourceJob = job.current;
    if (!active.current || sourceJob === null) {
      throw new Error("no regional-ground authority job is active");
    }
    const run = generation.current;
    clearPending();
    publish(run, { phase: "cancelling" });
    const request = beginRequest();
    try {
      const cancelled = await client.current.cancel(sourceJob, request.signal);
      if (current(run, request.id)) await processStatus.current?.(cancelled, run);
    } catch (reason) {
      if (current(run, request.id) && !publishNotFound(run, reason)) {
        failRequest(run, reason);
        throw reason;
      }
    }
  }, [beginRequest, clearPending, current, failRequest, publish, publishNotFound]);

  const reset = useCallback((): void => {
    if (active.current) {
      throw new Error("cannot reset an active or uncertain regional-ground authority job");
    }
    clearPending();
    generation.current += 1;
    active.current = false;
    submitted.current = false;
    job.current = null;
    setState(initialState());
  }, [clearPending]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      clearPending();
      generation.current += 1;
      active.current = false;
    };
  }, [clearPending]);

  const admitted = useMemo(() => {
    try {
      return qualifiedForExecution(
        parseRegionalGroundAuthorityCapability(options.capability),
      );
    } catch {
      return false;
    }
  }, [options.capability]);
  const controls = Object.freeze({
    submitEnabled: admitted && !active.current && !submitted.current,
    statusEnabled: active.current,
    cancelEnabled: active.current,
    resultEnabled: state.phase === "succeeded" && state.result !== null,
  });
  return { ...state, controls, submit, cancel, reconcile, reset };
}
