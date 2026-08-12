import { useCallback, useEffect, useRef, useState } from "react";

import type { MorrisAuthorityClient } from "../model/morrisAuthorityClient";
import type { MorrisAuthorityCapability } from "../model/morrisAuthorityCapability";
import type { MorrisJobEnvelope } from "../model/morrisAuthorityContract";
import {
  serializeMorrisAuthorityRequest,
  type MorrisAuthorityRequest,
} from "../model/morrisAuthorityRequest";

const DEFAULT_POLL_INTERVAL_MS = 500;

export interface MorrisAuthorityState {
  readonly capability: MorrisAuthorityCapability | null;
  readonly checking: boolean;
  readonly job: MorrisJobEnvelope | null;
  readonly error: string | null;
  readonly submitting: boolean;
}

interface ActiveOperation {
  readonly controller: AbortController;
  readonly generation: number;
}

interface JobIdentity {
  readonly requestId: string;
  readonly jobId: string;
}

const abortableDelay = (milliseconds: number, signal: AbortSignal): Promise<void> => (
  new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  })
);

const messageFrom = (error: unknown): string => (
  error instanceof Error ? error.message : "Unknown Morris authority error."
);

const isAbort = (error: unknown): boolean => (
  error instanceof DOMException && error.name === "AbortError"
);

const assertJobIdentity = (
  job: MorrisJobEnvelope, expected: JobIdentity, operation: "status" | "cancel",
): void => {
  if (job.requestId !== expected.requestId) {
    throw new Error(`Morris ${operation} response request identity mismatch.`);
  }
  if (job.jobId !== expected.jobId) {
    throw new Error(`Morris ${operation} response job identity mismatch.`);
  }
};

export function useMorrisAuthority(
  client: MorrisAuthorityClient | null,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
) {
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 0) {
    throw new RangeError("pollIntervalMs must be a nonnegative finite number");
  }
  const [state, setState] = useState<MorrisAuthorityState>({
    capability: null, checking: client !== null, job: null, error: null, submitting: false,
  });
  const generation = useRef(0);
  const operation = useRef<AbortController | null>(null);
  const inFlight = useRef(false);
  const jobIdentity = useRef<JobIdentity | null>(null);

  const beginOperation = useCallback(() => {
    operation.current?.abort();
    operation.current = new AbortController();
    generation.current += 1;
    return { controller: operation.current, generation: generation.current };
  }, []);

  useEffect(() => {
    if (client === null) {
      jobIdentity.current = null;
      setState({ capability: null, checking: false, job: null, error: null, submitting: false });
      return;
    }
    jobIdentity.current = null;
    const active = beginOperation();
    setState({ capability: null, checking: true, job: null, error: null, submitting: false });
    void client.capability(active.controller.signal).then((capability) => {
      if (generation.current !== active.generation) return;
      setState({ capability, checking: false, job: null, error: null, submitting: false });
    }).catch((error: unknown) => {
      if (generation.current !== active.generation || isAbort(error)) return;
      setState({ capability: null, checking: false, job: null, error: messageFrom(error), submitting: false });
    });
    return () => {
      generation.current += 1;
      operation.current?.abort();
      inFlight.current = false;
    };
  }, [beginOperation, client]);

  const pollUntilTerminal = useCallback(async (
    initial: MorrisJobEnvelope, active: ActiveOperation, identity: JobIdentity,
  ): Promise<MorrisJobEnvelope> => {
    if (client === null) return initial;
    let job = initial;
    assertJobIdentity(job, identity, "status");
    if (generation.current === active.generation) {
      setState((current) => ({ ...current, job, submitting: false }));
    }
    while (job.status === "queued" || job.status === "running") {
      await abortableDelay(pollIntervalMs, active.controller.signal);
      job = await client.status(identity.jobId, active.controller.signal);
      assertJobIdentity(job, identity, "status");
      if (generation.current !== active.generation) return job;
      setState((current) => ({ ...current, job, submitting: false }));
    }
    if (generation.current === active.generation) inFlight.current = false;
    return job;
  }, [client, pollIntervalMs]);

  const run = useCallback(async (request: MorrisAuthorityRequest): Promise<void> => {
    if (client === null || state.capability?.available !== true) {
      setState((current) => ({ ...current, error: "Morris authority is unavailable." }));
      return;
    }
    if (inFlight.current) return;
    inFlight.current = true;
    const active = beginOperation();
    jobIdentity.current = null;
    setState((current) => ({ ...current, job: null, error: null, submitting: true }));
    let document: ReturnType<typeof serializeMorrisAuthorityRequest>;
    try {
      document = serializeMorrisAuthorityRequest(request);
    } catch (error: unknown) {
      if (generation.current === active.generation) {
        inFlight.current = false;
        setState((current) => ({ ...current, error: messageFrom(error), submitting: false }));
      }
      return;
    }
    try {
      const job = await client.create(document, active.controller.signal);
      if (generation.current !== active.generation) return;
      if (job.requestId !== document.request_id) {
        throw new Error("Morris create response request identity mismatch.");
      }
      const identity = Object.freeze({ requestId: document.request_id, jobId: job.jobId });
      jobIdentity.current = identity;
      await pollUntilTerminal(job, active, identity);
    } catch (error: unknown) {
      if (generation.current !== active.generation || isAbort(error)) return;
      inFlight.current = false;
      setState((current) => ({ ...current, error: messageFrom(error), submitting: false }));
    }
  }, [beginOperation, client, pollUntilTerminal, state.capability]);

  const cancel = useCallback(async (): Promise<void> => {
    const jobId = state.job?.jobId;
    const identity = jobIdentity.current;
    if (client === null || jobId === undefined || identity === null) return;
    const active = beginOperation();
    try {
      const job = await client.cancel(identity.jobId, active.controller.signal);
      if (generation.current !== active.generation) return;
      assertJobIdentity(job, identity, "cancel");
      setState((current) => ({ ...current, job, error: null, submitting: false }));
      await pollUntilTerminal(job, active, identity);
    } catch (error: unknown) {
      if (generation.current !== active.generation || isAbort(error)) return;
      inFlight.current = false;
      setState((current) => ({ ...current, error: messageFrom(error), submitting: false }));
    }
  }, [beginOperation, client, pollUntilTerminal, state.job?.jobId]);

  const invalidate = useCallback((): void => {
    if (jobIdentity.current !== null || inFlight.current) {
      operation.current?.abort();
      operation.current = null;
      generation.current += 1;
    }
    jobIdentity.current = null;
    inFlight.current = false;
    setState((current) => ({ ...current, job: null, error: null, submitting: false }));
  }, []);

  return Object.freeze({ state, run, cancel, invalidate });
}
