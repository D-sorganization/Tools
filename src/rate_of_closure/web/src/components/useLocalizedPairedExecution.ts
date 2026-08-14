import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AttributionAuthorityTs } from "../model/localizedAttribution";
import type {
  LocalizedPairedProgressTs, LocalizedPairedRequestTs,
} from "../model/localizedAttributionExecution";
import {
  createLocalizedPairedExecutionService,
  type LocalizedPairedExecutionServiceTs,
} from "../model/localizedAttributionExecutionService";

export interface LocalizedPairedExecutionState {
  authority: AttributionAuthorityTs | null;
  status: string;
  busy: boolean;
  progress: LocalizedPairedProgressTs | null;
  run: (request: LocalizedPairedRequestTs) => Promise<void>;
  cancel: () => void;
  invalidate: (reason?: string) => void;
  reportFailure: (message: string) => void;
}

export function useLocalizedPairedExecution(
  serviceOverride?: LocalizedPairedExecutionServiceTs,
): LocalizedPairedExecutionState {
  const service = useMemo(() => serviceOverride ?? createLocalizedPairedExecutionService(),
    [serviceOverride]);
  const [authority, setAuthority] = useState<AttributionAuthorityTs | null>(null);
  const [status, setStatus] = useState("Ready: configure a separate paired study.");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<LocalizedPairedProgressTs | null>(null);
  const generation = useRef(0);
  const active = useRef<AbortController | null>(null);

  const invalidate = useCallback((reason = "Configuration changed; prior paired authority was cleared.") => {
    generation.current += 1;
    active.current?.abort(); active.current = null;
    setBusy(false); setProgress(null); setAuthority(null); setStatus(reason);
  }, []);

  const cancel = useCallback(() => {
    if (active.current === null) return;
    generation.current += 1;
    active.current.abort(); active.current = null;
    setBusy(false); setProgress(null);
    setStatus("Separate paired study cancelled. Prior paired authority was not replaced.");
  }, []);
  const reportFailure = useCallback((message: string) => {
    setStatus(`Cannot run paired study: ${message}. Prior paired authority was not replaced.`);
  }, []);

  const run = useCallback(async (request: LocalizedPairedRequestTs) => {
    const current = generation.current + 1;
    generation.current = current;
    active.current?.abort();
    const controller = new AbortController(); active.current = controller;
    setBusy(true); setProgress({ completedRuns: 0, totalRuns: request.interventionDeltasNm
      ? Object.keys(request.interventionDeltasNm).length * 2 : 0 });
    setStatus("Running separate paired study: 0 explicit trials completed.");
    try {
      const result = await service.execute(request, {
        signal: controller.signal,
        onProgress: (next) => {
          if (generation.current !== current || controller.signal.aborted) return;
          setProgress(next);
          setStatus(`Running separate paired study: ${next.completedRuns}/${next.totalRuns} explicit trials completed.`);
        },
      });
      if (generation.current !== current || controller.signal.aborted) return;
      setAuthority(result.authority);
      setStatus(`Paired study complete: ${result.authority.sources.length} sources, ${result.authority.pairs.length * 2} explicit trials. Planted-intervention response only; no causal inference.`);
    } catch (error) {
      if (generation.current !== current || controller.signal.aborted) return;
      setStatus(`Paired study failed before authority replacement: ${(error as Error).message}`);
      setProgress(null);
    } finally {
      if (generation.current === current) { active.current = null; setBusy(false); }
    }
  }, [service]);

  useEffect(() => () => {
    generation.current += 1; active.current?.abort(); active.current = null;
  }, []);
  return { authority, status, busy, progress, run, cancel, invalidate, reportFailure };
}
