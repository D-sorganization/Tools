import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";

const LOOPBACK_ORIGIN = /^http:\/\/127\.0\.0\.1:\d+$/;
const HARNESS_TIMEOUT_MS = 25_000;

interface HarnessEvent {
  readonly event: string;
  readonly [key: string]: unknown;
}

export interface CompanionHarness {
  readonly origin: string;
  command(command: "authority_hard_loss" | "gateway_hard_loss" |
    "inspect_public_exposure" | "observe_replacement"): Promise<HarnessEvent>;
  close(): Promise<void>;
}

function pythonCommand(): string {
  return process.env.PYTHON ?? (process.platform === "win32" ? "py" : "python");
}

function pythonArguments(mode: string): string[] {
  const prefix = process.platform === "win32" && process.env.PYTHON === undefined
    ? ["-3.11"] : [];
  return [...prefix, "-m", "tests.rate_of_closure.browser_companion_harness",
    "--authority-mode", mode];
}

function parseEvent(line: string): HarnessEvent {
  let value: unknown;
  try { value = JSON.parse(line); } catch {
    throw new Error("companion harness returned invalid metadata");
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("companion harness returned invalid metadata");
  }
  const event = Reflect.get(value, "event");
  if (typeof event !== "string") throw new Error("companion harness omitted event identity");
  return value as HarnessEvent;
}

function nextEvent(lines: readline.Interface,
  child: ChildProcessWithoutNullStreams): Promise<HarnessEvent> {
  if (child.exitCode !== null) {
    return Promise.reject(new Error("companion harness is not running"));
  }
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timer);
      lines.off("line", onLine);
      child.off("error", onError);
      child.off("exit", onExit);
    };
    const onLine = (line: string): void => {
      cleanup();
      try { resolve(parseEvent(line)); } catch (error) { reject(error); }
    };
    const onError = (): void => {
      cleanup();
      reject(new Error("companion harness process failed"));
    };
    const onExit = (): void => {
      cleanup();
      reject(new Error("companion harness exited before responding"));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("companion harness response timed out"));
    }, HARNESS_TIMEOUT_MS);
    lines.once("line", onLine);
    child.once("error", onError);
    child.once("exit", onExit);
  });
}

async function stopHarness(child: ChildProcessWithoutNullStreams,
  lines: readline.Interface, controlId: string): Promise<void> {
  if (child.exitCode !== null) {
    if (child.exitCode === 0) return;
    throw new Error("harness exited before shutdown");
  }
  const exited = new Promise<number | null>((resolve, reject) => {
    child.once("error", () => reject(new Error("harness failed during shutdown")));
    child.once("exit", resolve);
  });
  const response = nextEvent(lines, child);
  child.stdin.write(`${JSON.stringify({ command: "shutdown", control_id: controlId })}\n`);
  const timer = setTimeout(() => child.kill(), HARNESS_TIMEOUT_MS);
  let stopped: HarnessEvent;
  let code: number | null;
  try {
    [stopped, code] = await Promise.all([response, exited]);
  } finally {
    clearTimeout(timer);
  }
  if (stopped.event !== "stopped") throw new Error("companion harness rejected shutdown");
  if (code !== 0) throw new Error("harness failed during shutdown");
}

export async function startCompanionHarness(
  mode: "fast" | "cancellable" | "blocking" = "fast",
): Promise<CompanionHarness> {
  const repositoryRoot = path.resolve(import.meta.dirname, "../../../../../..");
  const inheritedPythonPath = process.env.PYTHONPATH;
  const pythonPath = [path.join(repositoryRoot, "src"), inheritedPythonPath]
    .filter((value): value is string => value !== undefined && value.length > 0)
    .join(path.delimiter);
  const child = spawn(pythonCommand(), pythonArguments(mode), {
    cwd: repositoryRoot,
    env: { ...process.env, PYTHONPATH: pythonPath },
    stdio: ["pipe", "pipe", "pipe"],
    windowsHide: true,
  });
  // Drain native diagnostics without retaining or publishing potentially
  // sensitive process context in the browser qualification artifact.
  child.stderr.resume();
  const lines = readline.createInterface({ input: child.stdout });
  let ready: HarnessEvent;
  try {
    ready = await nextEvent(lines, child);
  } catch (error) {
    child.kill();
    lines.close();
    throw error;
  }
  const origin = ready.gateway_url;
  const controlId = ready.control_id;
  if (ready.event !== "ready" || typeof origin !== "string" ||
      !LOOPBACK_ORIGIN.test(origin.replace(/\/$/, "")) || typeof controlId !== "string" ||
      controlId.length === 0 || controlId.length > 128 || controlId.trim() !== controlId) {
    child.kill();
    throw new Error("companion harness returned unsafe readiness metadata");
  }
  return {
    origin: origin.replace(/\/$/, ""),
    async command(command): Promise<HarnessEvent> {
      const response = nextEvent(lines, child);
      child.stdin.write(`${JSON.stringify({ command, control_id: controlId })}\n`);
      return await response;
    },
    close: () => stopHarness(child, lines, controlId),
  };
}
