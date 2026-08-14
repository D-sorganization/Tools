import { useRef, type KeyboardEvent, type RefObject } from "react";

import {
  APP_COMMAND_ID,
  commandsInGroup,
  type AppCommandId,
} from "../model/appCommands";
import type { AppTheme } from "../model/appTheme";
import type { RegionalGroundVariationRequestPort } from "../model/regionalGroundVariationWorkspace";
import type { RegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import {
  PRIMARY_VIEWS,
  restorePrimaryViewDefaults,
  setPrimaryViewVisibility,
  shiftPrimaryView,
  type PrimaryViewState,
} from "../model/viewPreferences";
import { RegionalGroundVariationFileCommands } from "./RegionalGroundVariationFileCommands";
import { RegionalGroundExecutionFileCommands } from "./RegionalGroundExecutionFileCommands";
import { useViewportClampedPopover } from "./useViewportClampedPopover";

interface AppToolstripProps {
  readonly moduleState: PrimaryViewState;
  readonly theme: AppTheme;
  readonly shortcutHelpOpen: boolean;
  readonly onModuleStateChange: (state: PrimaryViewState) => void;
  readonly onCommand: (command: AppCommandId) => void;
  readonly onShortcutHelpOpenChange: (open: boolean) => void;
  /** Typed seam for the contextual regional-ground File commands. */
  readonly regionalGroundVariationRequestPort?: RegionalGroundVariationRequestPort;
  readonly regionalGroundExecutionWorkspace?: RegionalGroundExecutionWorkspace;
  readonly fileStatus?: string;
  readonly fileError?: string | null;
}

interface ToolCommandDescriptor {
  readonly id: AppCommandId;
  readonly label: string;
  readonly shortcut: string;
  readonly accessibleName: string;
}

const MENU_CLASS =
  "relative shrink-0 rounded-lg border border-slate-700/80 bg-slate-900/90 text-sm text-slate-200";
const SUMMARY_CLASS =
  "cursor-pointer list-none rounded-lg px-3 py-2 font-semibold hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400";
const POPOVER_CLASS =
  "absolute left-0 z-40 mt-1 max-w-[calc(100vw-2rem)] rounded-xl border border-slate-700 bg-slate-950 p-3 shadow-2xl shadow-black/50";
const COMMAND_CLASS =
  "w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-45";
const FILE_COMMAND_HINTS: Partial<Record<AppCommandId, string>> = {
  [APP_COMMAND_ID.fileNewWorkspace]: "Reset supported inputs and views to a clean workspace.",
  [APP_COMMAND_ID.fileOpenWorkspace]: "Choose and validate a whole workspace before applying it.",
  [APP_COMMAND_ID.fileSaveWorkspaceAs]: "Download a new whole-workspace JSON copy.",
  [APP_COMMAND_ID.fileImportWorkspace]: "Import a strict cross-client view-layout JSON file.",
  [APP_COMMAND_ID.fileExportWorkspace]: "Download the current cross-client view layout.",
  [APP_COMMAND_ID.fileCloseWorkspace]: "Close the current session and load clean defaults.",
};

const commandShortcut = (id: AppCommandId): string | undefined =>
  commandsInGroup("global").find((command) => command.id === id)?.shortcut;

const requireCommandShortcut = (id: AppCommandId): string => {
  const shortcut = commandShortcut(id);
  if (shortcut === undefined || shortcut.length === 0) {
    throw new TypeError(`Global command ${id} requires a visible shortcut.`);
  }
  return shortcut;
};

const popoverClass = (widthClass = "w-64") => `${POPOVER_CLASS} ${widthClass}`;

const toolCommands = (theme: AppTheme): readonly ToolCommandDescriptor[] => [
  {
    id: APP_COMMAND_ID.globalOpenGlossary,
    label: "Open Glossary",
    shortcut: requireCommandShortcut(APP_COMMAND_ID.globalOpenGlossary),
    accessibleName: "Open Glossary",
  },
  {
    id: APP_COMMAND_ID.globalToggleTheme,
    label: `Theme: ${theme === "dark" ? "Dark" : "Light"}`,
    shortcut: requireCommandShortcut(APP_COMMAND_ID.globalToggleTheme),
    accessibleName: `Toggle Theme; currently ${theme}`,
  },
  {
    id: APP_COMMAND_ID.globalShowShortcuts,
    label: "Keyboard Shortcuts",
    shortcut: requireCommandShortcut(APP_COMMAND_ID.globalShowShortcuts),
    accessibleName: "Keyboard Shortcuts",
  },
  {
    id: APP_COMMAND_ID.globalOpenCurrentModuleHelp,
    label: "Current Module Help",
    shortcut: requireCommandShortcut(APP_COMMAND_ID.globalOpenCurrentModuleHelp),
    accessibleName: "Current Module Help",
  },
];

function ShortcutDialog({ onClose }: { readonly onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const close = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "Tab") {
      event.preventDefault();
      closeButton.current?.focus();
    }
  };
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcut-dialog-title"
      onKeyDown={close}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4"
    >
      <section className="w-full max-w-lg rounded-2xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
        <div className="flex items-center justify-between gap-4">
          <h2 id="shortcut-dialog-title" className="text-lg font-bold text-slate-100">
            Keyboard Shortcuts
          </h2>
          <button
            ref={closeButton}
            autoFocus
            type="button"
            aria-label="Close Keyboard Shortcuts"
            onClick={onClose}
            className="rounded-md px-3 py-2 text-slate-300 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            Close
          </button>
        </div>
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm">
          {commandsInGroup("global").filter(({ shortcut }) => shortcut).map((command) => (
            <div key={command.id} className="contents">
              <dt><kbd className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-sky-200">{command.shortcut}</kbd></dt>
              <dd className="text-slate-300">{command.label}</dd>
            </div>
          ))}
          <div className="contents">
            <dt><kbd className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-sky-200">Arrow keys</kbd></dt>
            <dd className="text-slate-300">Move focus between visible workspace tabs</dd>
          </div>
          <div className="contents">
            <dt><kbd className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-sky-200">Alt+← / Alt+→</kbd></dt>
            <dd className="text-slate-300">Reorder the focused workspace tab</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}

const CONTEXTUAL_FILE_COMMANDS: readonly AppCommandId[] = [
  APP_COMMAND_ID.fileOpenRegionalGroundVariationRequest,
  APP_COMMAND_ID.fileSaveRegionalGroundVariationRequestAs,
  APP_COMMAND_ID.fileOpenRegionalGroundExecutionJob,
  APP_COMMAND_ID.fileSaveRegionalGroundExecutionJobAs,
  APP_COMMAND_ID.fileSaveRegionalGroundExecutionResultAs,
  APP_COMMAND_ID.fileExportRegionalGroundExecutionRowsCsv,
];

function FileMenu({
  active,
  requestPort: variationRequestPort,
  executionWorkspace: groundExecutionWorkspace,
  onCommand,
}: {
  readonly active: PrimaryViewState["active"];
  readonly requestPort?: RegionalGroundVariationRequestPort;
  readonly executionWorkspace?: RegionalGroundExecutionWorkspace;
  readonly onCommand: (command: AppCommandId) => void;
}) {
  const fileCommands = commandsInGroup("file").filter(
    ({ id }) => !CONTEXTUAL_FILE_COMMANDS.includes(id),
  );
  const contextual =
    variationRequestPort !== undefined &&
    (active === "variation" || active === "regional-surfaces");
  const requestPort = contextual ? variationRequestPort : undefined;
  const executionWorkspace =
    active === "ground-playback" ? groundExecutionWorkspace : undefined;
  const popover = useViewportClampedPopover();
  return (
    <details className={MENU_CLASS} onToggle={popover.onToggle}>
      <summary className={SUMMARY_CLASS}>File</summary>
      <div
        ref={popover.panelRef}
        style={popover.style}
        className={popoverClass()}
        role="group"
        aria-label="File commands"
      >
        {requestPort !== undefined && (
          <RegionalGroundVariationFileCommands
            requestPort={requestPort}
            onCommand={onCommand}
            commandClassName={COMMAND_CLASS}
          />
        )}
        {executionWorkspace !== undefined && (
          <RegionalGroundExecutionFileCommands
            workspace={executionWorkspace}
            onCommand={onCommand}
            commandClassName={COMMAND_CLASS}
          />
        )}
        {fileCommands.map((command) => (
          <button
            key={command.id}
            type="button"
            disabled={!command.enabled}
            data-command-id={command.id}
            title={command.disabledReason ?? FILE_COMMAND_HINTS[command.id]}
            onClick={() => onCommand(command.id)}
            className={COMMAND_CLASS}
          >
            {command.label}
          </button>
        ))}
        {fileCommands.filter(({ disabledReason }) => disabledReason !== null).map((command) => (
          <p key={`${command.id}-reason`}
            className="mt-2 border-t border-slate-800 pt-2 text-xs leading-relaxed text-amber-200">
            {command.label}: {command.disabledReason}
          </p>
        ))}
      </div>
    </details>
  );
}

function ModuleRow({
  module,
  state,
  onChange,
}: {
  readonly module: (typeof PRIMARY_VIEWS)[number];
  readonly state: PrimaryViewState;
  readonly onChange: (state: PrimaryViewState) => void;
}) {
  const index = state.order.indexOf(module.id);
  const reorder = (offset: -1 | 1) => onChange({
    ...state,
    order: shiftPrimaryView(state.order, module.id, offset),
  });
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-2 rounded-md px-2 py-1 hover:bg-slate-900">
      <label
        className="flex min-w-0 items-center gap-2"
        title={module.required
          ? `${module.label} is required so the workspace always has a safe fallback.`
          : `Show or hide the ${module.label} workspace module.`}
      >
        <input
          type="checkbox"
          checked={state.visible.includes(module.id)}
          disabled={module.required}
          onChange={(event) => onChange(
            setPrimaryViewVisibility(state, module.id, event.target.checked),
          )}
          className="size-4 accent-sky-500"
        />
        <span className="truncate">{module.label}</span>
        {module.required && <span className="text-[10px] uppercase text-slate-500">Required</span>}
      </label>
      <span className="flex gap-1">
        {([-1, 1] as const).map((offset) => (
          <button
            key={offset}
            type="button"
            aria-label={`Move ${module.label} ${offset < 0 ? "up" : "down"}`}
            disabled={offset < 0 ? index === 0 : index === state.order.length - 1}
            onClick={() => reorder(offset)}
            className="rounded px-2 py-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100 disabled:opacity-30"
          >
            {offset < 0 ? "↑" : "↓"}
          </button>
        ))}
      </span>
    </div>
  );
}

function ViewMenu({
  state,
  onChange,
  onCommand,
}: {
  readonly state: PrimaryViewState;
  readonly onChange: (state: PrimaryViewState) => void;
  readonly onCommand: (command: AppCommandId) => void;
}) {
  const popover = useViewportClampedPopover();
  const orderedModules = state.order.map((id) =>
    PRIMARY_VIEWS.find((module) => module.id === id),
  ).filter((module): module is (typeof PRIMARY_VIEWS)[number] => module !== undefined);
  return (
    <details className={MENU_CLASS} onToggle={popover.onToggle}>
      <summary
        data-command-id={APP_COMMAND_ID.viewManageModules}
        onClick={() => onCommand(APP_COMMAND_ID.viewManageModules)}
        className={SUMMARY_CLASS}
      >
        View
      </summary>
      <div
        ref={popover.panelRef}
        style={popover.style}
        className={popoverClass("w-96")}
        role="group"
        aria-label="Workspace modules"
      >
        <p className="mb-2 text-xs text-slate-400">Show, hide, or reorder workspace modules.</p>
        {orderedModules.map((module) => (
          <ModuleRow key={module.id} module={module} state={state} onChange={onChange} />
        ))}
        <button
          type="button"
          data-command-id={APP_COMMAND_ID.viewRestoreDefaultWorkspace}
          title="Restore the default module order, visibility, and active view."
          onClick={() => {
            onCommand(APP_COMMAND_ID.viewRestoreDefaultWorkspace);
            onChange(restorePrimaryViewDefaults());
          }}
          className={`${COMMAND_CLASS} mt-2 border-t border-slate-800`}
        >
          Restore default modules
        </button>
      </div>
    </details>
  );
}

function ToolsMenu({
  theme,
  shortcutTrigger,
  run,
}: {
  readonly theme: AppTheme;
  readonly shortcutTrigger: RefObject<HTMLButtonElement>;
  readonly run: (id: AppCommandId) => void;
}) {
  const popover = useViewportClampedPopover();
  const commands = toolCommands(theme);
  return (
    <details className={MENU_CLASS} onToggle={popover.onToggle}>
      <summary className={SUMMARY_CLASS}>Tools</summary>
      <div
        ref={popover.panelRef}
        style={popover.style}
        className={popoverClass()}
        role="group"
        aria-label="Global tools"
      >
        {commands.map((command) => (
          <button
            key={command.id}
            ref={command.id === APP_COMMAND_ID.globalShowShortcuts
              ? shortcutTrigger
              : undefined}
            type="button"
            data-command-id={command.id}
            aria-keyshortcuts={commandShortcut(command.id)}
            aria-label={command.accessibleName}
            onClick={() => run(command.id)}
            className={COMMAND_CLASS}
          >
            {command.label}
            <span className="float-right text-xs text-slate-500">{command.shortcut}</span>
          </button>
        ))}
      </div>
    </details>
  );
}

export function AppToolstrip({
  moduleState,
  theme,
  shortcutHelpOpen,
  onModuleStateChange,
  onCommand,
  onShortcutHelpOpenChange,
  regionalGroundVariationRequestPort,
  regionalGroundExecutionWorkspace,
  fileStatus = "Workspace ready",
  fileError = null,
}: AppToolstripProps) {
  const shortcutTrigger = useRef<HTMLButtonElement>(null);
  const run = (id: AppCommandId) => {
    onCommand(id);
    if (id === APP_COMMAND_ID.globalShowShortcuts) onShortcutHelpOpenChange(true);
  };
  const closeShortcuts = () => {
    onShortcutHelpOpenChange(false);
    queueMicrotask(() => shortcutTrigger.current?.focus());
  };
  return (
    <>
      <div
        role="toolbar"
        aria-label="Application commands"
        className="sticky top-0 z-30 mb-5 overflow-visible rounded-xl border border-slate-700/80 bg-slate-950/90 p-2 shadow-xl shadow-black/30 backdrop-blur"
      >
        <div className="flex flex-wrap items-start gap-2">
          <FileMenu
            active={moduleState.active}
            requestPort={regionalGroundVariationRequestPort}
            executionWorkspace={regionalGroundExecutionWorkspace}
            onCommand={run}
          />
          <ViewMenu state={moduleState} onChange={onModuleStateChange} onCommand={onCommand} />
          <div className="flex max-w-full shrink-0 items-stretch gap-1 overflow-x-auto rounded-lg border border-slate-700/80 bg-slate-900/90 p-1">
            {([
              [APP_COMMAND_ID.viewShowImpact, "Impact"],
              [APP_COMMAND_ID.viewShowSwing, "Swing"],
              [APP_COMMAND_ID.viewShowFlight, "Flight"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                data-command-id={id}
                title={`Show the ${label.toLowerCase()} view in the main workspace.`}
                onClick={() => run(id)}
                className="rounded-md px-3 py-1.5 text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                {label}
              </button>
            ))}
          </div>
          <ToolsMenu theme={theme} shortcutTrigger={shortcutTrigger} run={run} />
          <span role={fileError === null ? "status" : "alert"}
            className={`self-center px-2 text-xs ${fileError === null ? "text-slate-400" : "text-rose-300"}`}>
            {fileError ?? fileStatus}
          </span>
        </div>
      </div>
      {shortcutHelpOpen && <ShortcutDialog onClose={closeShortcuts} />}
    </>
  );
}
