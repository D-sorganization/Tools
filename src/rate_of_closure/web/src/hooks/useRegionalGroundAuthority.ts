/** Lifecycle-safe capability polling for the local Python ground authority. */

import { useEffect, useMemo, useState } from "react";

import {
  fetchRegionalGroundAuthorityCapability,
  parseRegionalGroundAuthorityCapability,
  staticInspectionRegionalGroundCapability,
  unavailableRegionalGroundAuthorityCapability,
  type RegionalGroundAuthorityCapability,
} from "../model/regionalGroundAuthority";
import {
  LOCAL_COMPANION_WEB_RUNTIME,
  type WebRuntime,
} from "../model/webRuntime";

const MIN_POLL_INTERVAL_MS = 250;
const DEFAULT_POLL_INTERVAL_MS = 5_000;

export interface RegionalGroundExecutionControlState {
  readonly submitEnabled: boolean;
  readonly statusEnabled: boolean;
  readonly cancelEnabled: boolean;
  readonly resultEnabled: boolean;
}

export interface RegionalGroundAuthorityState {
  readonly capability: RegionalGroundAuthorityCapability;
  readonly checking: boolean;
  readonly controls: RegionalGroundExecutionControlState;
}

export type RegionalGroundAuthorityQuery = (
  signal: AbortSignal,
) => Promise<RegionalGroundAuthorityCapability>;

export interface RegionalGroundAuthorityOptions {
  readonly query?: RegionalGroundAuthorityQuery;
  readonly pollIntervalMs?: number;
  readonly runtime?: WebRuntime;
}

const defaultQuery: RegionalGroundAuthorityQuery = (signal) =>
  fetchRegionalGroundAuthorityCapability(fetch, signal);

const initialCapability = (): RegionalGroundAuthorityCapability =>
  unavailableRegionalGroundAuthorityCapability(
    "authority_unreachable",
    "Local Python execution authority has not been checked.",
  );

const controlsFor = (
  capability: RegionalGroundAuthorityCapability,
): RegionalGroundExecutionControlState => Object.freeze({
  submitEnabled: capability.available && capability.regional_ground_execution,
  statusEnabled: false,
  cancelEnabled: false,
  resultEnabled: false,
});

const validatePollInterval = (value: number): number => {
  if (!Number.isSafeInteger(value) || value < MIN_POLL_INTERVAL_MS) {
    throw new RangeError(`pollIntervalMs must be an integer >= ${MIN_POLL_INTERVAL_MS}`);
  }
  return value;
};

/** Poll capability without overlapping requests or publishing obsolete state. */
export function useRegionalGroundAuthority(
  options: RegionalGroundAuthorityOptions = {},
): RegionalGroundAuthorityState {
  const query = options.query ?? defaultQuery;
  const runtime = options.runtime ?? LOCAL_COMPANION_WEB_RUNTIME;
  const interval = validatePollInterval(
    options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS,
  );
  const [capability, setCapability] = useState(initialCapability);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    if (runtime.mode === "static_inspection") {
      setCapability(staticInspectionRegionalGroundCapability());
      setChecking(false);
      return;
    }
    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController | undefined;
    const poll = async (): Promise<void> => {
      controller = new AbortController();
      const next = await query(controller.signal)
        .then(parseRegionalGroundAuthorityCapability)
        .catch(() => initialCapability());
      if (disposed) return;
      setCapability(next);
      setChecking(false);
      timer = window.setTimeout(() => { void poll(); }, interval);
    };
    void poll();
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
      controller?.abort();
    };
  }, [interval, query, runtime.mode]);

  const controls = useMemo(() => controlsFor(capability), [capability]);
  return { capability, checking, controls };
}
