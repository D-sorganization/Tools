import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import type { CapabilityRunOutput } from "../model/capabilityRun";
import {
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowInputs,
  defaultCapabilityWorkflowInputs,
  overlayCapabilityWorkflowInputs,
  type CapabilityWorkflowDocument,
  type CapabilityWorkflowInputs,
} from "../model/capabilityWorkflow";
import {
  runCapabilityInWorker,
  type CapabilityRunController,
  type CapabilityRunner,
} from "../model/capabilityWorkerClient";

export interface CapabilityOptimizationState {
  readonly inputs: CapabilityWorkflowInputs;
  readonly output: CapabilityRunOutput | null;
  readonly status: string;
  readonly error: string | null;
  readonly progress: { readonly completed: number; readonly total: number };
  readonly running: boolean;
  document: () => CapabilityWorkflowDocument;
  update: (key: keyof CapabilityWorkflowInputs, value: string | number) => void;
  run: () => void;
  cancel: () => void;
  load: (file: File) => Promise<void>;
}

export interface CapabilityWorkflowAuthority {
  readonly workflow: CapabilityWorkflowDocument;
  readonly onWorkflowChange: Dispatch<SetStateAction<CapabilityWorkflowDocument>>;
}

const defaultWorkflow = (): CapabilityWorkflowDocument =>
  buildCapabilityWorkflow(defaultCapabilityWorkflowInputs());
const message = (reason: unknown): string =>
  reason instanceof Error ? reason.message : String(reason);

export function useCapabilityOptimization(
  runner: CapabilityRunner = runCapabilityInWorker,
  authority?: CapabilityWorkflowAuthority,
): CapabilityOptimizationState {
  const [localWorkflow, setLocalWorkflow] = useState(defaultWorkflow);
  const workflow = authority?.workflow ?? localWorkflow;
  const setWorkflow = authority?.onWorkflowChange ?? setLocalWorkflow;
  const [inputs, setInputs] = useState(() => capabilityWorkflowInputs(workflow));
  const [output, setOutput] = useState<CapabilityRunOutput | null>(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const active = useRef<CapabilityRunController | null>(null);
  const runId = useRef(0);
  const priorWorkflow = useRef(workflow);
  const invalidate = (next: string): void => {
    runId.current += 1;
    active.current?.cancel();
    active.current = null;
    setRunning(false);
    setOutput(null);
    setError(null);
    setStatus(next);
    setProgress({ completed: 0, total: 0 });
  };
  useEffect(() => () => { runId.current += 1; active.current?.cancel(); }, []);
  useLayoutEffect(() => {
    if (priorWorkflow.current !== workflow) {
      setInputs(capabilityWorkflowInputs(workflow));
      invalidate("Inputs changed — run again");
    }
    priorWorkflow.current = workflow;
  }, [workflow]);
  const document = (): CapabilityWorkflowDocument =>
    overlayCapabilityWorkflowInputs(workflow, inputs);
  const update = (key: keyof CapabilityWorkflowInputs, value: string | number): void => {
    invalidate("Inputs changed — run again");
    const next = { ...inputs, [key]: value };
    setInputs(next);
    try { setWorkflow(overlayCapabilityWorkflowInputs(workflow, next)); }
    catch { /* Keep an invalid draft local until validation or correction. */ }
  };
  const run = (): void => {
    invalidate("Validating calculation basis");
    const currentRun = ++runId.current;
    try {
      const current = document();
      setProgress({ completed: 0,
        total: current.request.candidateBudget * current.request.ensembleSize });
      setStatus("Running in background");
      const controller = runner(current, (next) => {
        if (currentRun === runId.current) setProgress(next);
      });
      active.current = controller;
      setRunning(true);
      void controller.promise.then((result) => {
        if (currentRun !== runId.current) return;
        active.current = null;
        setRunning(false);
        setOutput(result);
        setStatus("Completed");
      }).catch((reason: unknown) => {
        if (currentRun !== runId.current) return;
        active.current = null;
        setRunning(false);
        setStatus("Failed");
        setError(message(reason));
      });
    } catch (reason: unknown) {
      setStatus("Invalid inputs");
      setError(message(reason));
    }
  };
  const load = async (file: File): Promise<void> => {
    try {
      const parsed = capabilityWorkflowFromJson(await file.text());
      capabilityWorkflowInputs(parsed);
      setWorkflow(parsed);
      setInputs(capabilityWorkflowInputs(parsed));
      invalidate("Workflow loaded — run when ready");
    } catch (reason: unknown) { setError(message(reason)); }
  };
  return { inputs, output, status, error, progress, running, document,
    update, run, cancel: () => invalidate("Cancelled"), load };
}
