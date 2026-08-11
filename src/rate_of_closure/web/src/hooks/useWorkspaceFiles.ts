import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type RefObject,
} from "react";

import { APP_COMMAND_ID, type AppCommandId } from "../model/appCommands";
import {
  exportViewWorkspace,
  importViewWorkspace,
  type ViewWorkspace,
} from "../model/viewWorkspace";
import type { WorkspaceSessionSnapshot } from "../model/workspaceSession";

type PickerMode = "workspace" | "view";
const APP_VERSION = "1.14.34";

interface WorkspaceFileOptions {
  readonly snapshot: WorkspaceSessionSnapshot;
  readonly initialSnapshot: WorkspaceSessionSnapshot;
  readonly applySnapshot: (snapshot: WorkspaceSessionSnapshot) => void;
  readonly applyViewWorkspace: (workspace: ViewWorkspace) => void;
}

export interface WorkspaceFileController {
  readonly fileInputRef: RefObject<HTMLInputElement>;
  readonly fileAccept: string;
  readonly status: string;
  readonly error: string | null;
  readonly dirty: boolean;
  readonly handleCommand: (command: AppCommandId) => boolean;
  readonly onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
}

function fingerprint(snapshot: WorkspaceSessionSnapshot): string {
  return JSON.stringify(snapshot);
}

function metadata(title: string) {
  const now = new Date().toISOString();
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}`;
  return {
    documentId: `workspace.${random}`,
    title,
    createdAtUtc: now,
    modifiedAtUtc: now,
    appVersion: APP_VERSION,
  };
}

function downloadText(text: string, filename: string): void {
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(new Error("The selected file could not be read."));
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("The selected file is not UTF-8 text."));
    reader.readAsText(file, "UTF-8");
  });
}

export function useWorkspaceFiles(
  options: WorkspaceFileOptions,
): WorkspaceFileController {
  const input = useRef<HTMLInputElement>(null);
  const mode = useRef<PickerMode>("workspace");
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const [baseline, setBaseline] = useState(() =>
    fingerprint(options.initialSnapshot),
  );
  const [status, setStatus] = useState("Workspace ready");
  const [error, setError] = useState<string | null>(null);
  const currentFingerprint = useMemo(
    () => fingerprint(options.snapshot),
    [options.snapshot],
  );
  const dirty = currentFingerprint !== baseline;
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;

  const confirmDiscard = useCallback(
    (action: string): boolean =>
      !dirtyRef.current ||
      window.confirm(`Discard unsaved workspace changes and ${action}?`),
    [],
  );

  const reset = useCallback(
    (label: string) => {
      if (!confirmDiscard(label.toLowerCase())) return;
      options.applySnapshot(options.initialSnapshot);
      setBaseline(fingerprint(options.initialSnapshot));
      setError(null);
      setStatus(label);
    },
    [confirmDiscard, options],
  );

  const choose = useCallback((nextMode: PickerMode) => {
    mode.current = nextMode;
    input.current?.click();
  }, []);

  const handleCommand = useCallback(
    (command: AppCommandId): boolean => {
      setError(null);
      if (command === APP_COMMAND_ID.fileNewWorkspace) {
        reset("New workspace created");
        return true;
      }
      if (command === APP_COMMAND_ID.fileCloseWorkspace) {
        reset("Workspace closed; clean defaults loaded");
        return true;
      }
      if (command === APP_COMMAND_ID.fileOpenWorkspace) {
        choose("workspace");
        return true;
      }
      if (command === APP_COMMAND_ID.fileImportWorkspace) {
        choose("view");
        return true;
      }
      if (command === APP_COMMAND_ID.fileSaveWorkspaceAs) {
        void import("../model/workspaceSession")
          .then(({ createWorkspaceDocument }) => {
            downloadText(
              createWorkspaceDocument(
                options.snapshot,
                metadata("Rate workspace"),
              ),
              "rate-of-closure.roc-workspace.json",
            );
            setBaseline(currentFingerprint);
            setStatus("Workspace copy downloaded");
          })
          .catch((caught: unknown) => {
            setError(
              caught instanceof Error
                ? caught.message
                : "Workspace download failed",
            );
          });
        return true;
      }
      if (command === APP_COMMAND_ID.fileExportWorkspace) {
        try {
          downloadText(
            exportViewWorkspace(options.snapshot.viewWorkspace),
            "layout.roc-view.json",
          );
          setStatus("View layout downloaded");
        } catch (caught) {
          setError(
            caught instanceof Error ? caught.message : "View export failed",
          );
        }
        return true;
      }
      return (
        command === APP_COMMAND_ID.fileSaveWorkspace ||
        command === APP_COMMAND_ID.fileOpenRecentWorkspace
      );
    },
    [choose, currentFingerprint, options.snapshot, reset],
  );

  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];
      event.currentTarget.value = "";
      if (file === undefined) return;
      void readFileText(file)
        .then((text) => {
          try {
            if (mode.current === "view") {
              const parsed = importViewWorkspace(text);
              if (!confirmDiscard("import the selected view layout")) return;
              optionsRef.current.applyViewWorkspace(parsed);
              setStatus(`Imported ${file.name}`);
              setError(null);
              return;
            }
            void import("../model/workspaceSession")
              .then(({ parseWorkspaceDocument }) => {
                const current = optionsRef.current;
                const parsed = parseWorkspaceDocument(text, {
                  legacySimulationFallback: current.snapshot.simulation,
                  legacyTorqueFallback: current.snapshot.torque,
                  legacyVariationFallback: current.snapshot.variation,
                });
                if (!confirmDiscard("open the selected workspace")) return;
                current.applySnapshot(parsed);
                setBaseline(fingerprint(parsed));
                setStatus(`Opened ${file.name}`);
                setError(null);
              })
              .catch((caught: unknown) => {
                setError(
                  caught instanceof Error
                    ? caught.message
                    : "Workspace file is invalid",
                );
              });
          } catch (caught) {
            setError(
              caught instanceof Error
                ? caught.message
                : "Workspace file is invalid",
            );
          }
        })
        .catch((caught: unknown) => {
          setError(
            caught instanceof Error
              ? caught.message
              : "Workspace file read failed",
          );
        });
    },
    [confirmDiscard],
  );

  return {
    fileInputRef: input,
    fileAccept: ".roc-workspace.json,.roc-view.json,.json",
    status,
    error,
    dirty,
    handleCommand,
    onFileChange,
  };
}
