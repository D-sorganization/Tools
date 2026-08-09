import { useEffect, useRef, useState } from "react";

import type { CapabilityRunOutput } from "../model/capabilityRun";
import {
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowInputs,
  defaultCapabilityWorkflowInputs,
  type CapabilityWorkflowInputs,
} from "../model/capabilityWorkflow";
import {
  runCapabilityInWorker,
  type CapabilityRunController,
  type CapabilityRunner,
} from "../model/capabilityWorkerClient";

export interface CapabilityOptimizationState {
  readonly inputs: CapabilityWorkflowInputs; readonly output: CapabilityRunOutput | null;
  readonly status: string; readonly error: string | null;
  readonly progress: { readonly completed: number; readonly total: number };
  readonly running: boolean;
  update: (key: keyof CapabilityWorkflowInputs, value: string | number) => void;
  run: () => void; cancel: () => void; load: (file: File) => Promise<void>;
}

const message = (reason: unknown): string =>
  reason instanceof Error ? reason.message : String(reason);

export function useCapabilityOptimization(
  runner: CapabilityRunner = runCapabilityInWorker,
): CapabilityOptimizationState {
  const [inputs, setInputs] = useState(defaultCapabilityWorkflowInputs);
  const [output, setOutput] = useState<CapabilityRunOutput | null>(null);
  const [status, setStatus] = useState("Ready"); const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const active = useRef<CapabilityRunController | null>(null); const runId = useRef(0);
  const invalidate = (next: string): void => {
    runId.current += 1; active.current?.cancel(); active.current = null;
    setRunning(false);
    setOutput(null); setError(null); setStatus(next); setProgress({ completed: 0, total: 0 });
  };
  useEffect(() => () => { runId.current += 1; active.current?.cancel(); }, []);
  const update = (key: keyof CapabilityWorkflowInputs, value: string | number): void => {
    invalidate("Inputs changed — run again"); setInputs((current) => ({ ...current, [key]: value }));
  };
  const run = (): void => {
    invalidate("Validating calculation basis"); const currentRun = ++runId.current;
    try {
      const document = buildCapabilityWorkflow(inputs);
      setProgress({ completed: 0, total: document.request.candidateBudget * document.request.ensembleSize });
      setStatus("Running in background"); const controller = runner(document, (next) => {
        if (currentRun === runId.current) setProgress(next);
      });
      active.current = controller; setRunning(true); void controller.promise.then((result) => {
        if (currentRun !== runId.current) return;
        active.current = null; setRunning(false); setOutput(result); setStatus("Completed");
      }).catch((reason: unknown) => {
        if (currentRun !== runId.current) return;
        active.current = null; setRunning(false); setStatus("Failed"); setError(message(reason));
      });
    } catch (reason: unknown) { setStatus("Invalid inputs"); setError(message(reason)); }
  };
  const load = async (file: File): Promise<void> => {
    try { setInputs(capabilityWorkflowInputs(capabilityWorkflowFromJson(await file.text())));
      invalidate("Workflow loaded — run when ready"); }
    catch (reason: unknown) { setError(message(reason)); }
  };
  return { inputs, output, status, error, progress, running,
    update, run, cancel: () => invalidate("Cancelled"), load };
}
