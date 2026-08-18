import { useRef, type KeyboardEvent } from "react";

import {
  APP_COMMAND_ID,
  commandsInGroup,
  type AppCommandId,
} from "../model/appCommands";
import type { AppTheme } from "../model/appTheme";
import {
  PRIMARY_VIEWS,
  restorePrimaryViewDefaults,
  setPrimaryViewVisibility,
  shiftPrimaryView,
  type PrimaryViewState,
} from "../model/viewPreferences";

interface AppToolstripProps {
  readonly moduleState: PrimaryViewState;
  readonly theme: AppTheme;
  readonly shortcutHelpOpen: boolean;
  readonly onModuleStateChange: (state: PrimaryViewState) => void;
  readonly onCommand: (command: AppCommandId) => void;
  readonly onShortcutHelpOpenChange: (open: boolean) => void;
}

const MENU_CLASS =
  "relative shrink-0 rounded-lg border border-slate-700/80 bg-slate-900/90 text-sm text-slate-200";
const SUMMARY_CLASS =
  "cursor-pointer list-none rounded-lg px-2 py-2 font-semibold hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 sm:px-3";
const POPOVER_CLASS =
  "absolute left-0 z-40 mt-1 min-w-64 rounded-xl border border-slate-700 bg-slate-950 p-3 shadow-2xl shadow-black/50";
const COMMAND_CLASS =
  "w-full rounded-md px-3 py-2 text-left text-sm hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-45";

const commandShortcut = (id: AppCommandId): string | undefined =>
  commandsInGroup("global").find((command) => command.id === id)?.shortcut;

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

function FileMenu() {
  const fileCommands = commandsInGroup("file");
  return (
    <details className={MENU_CLASS}>
      <summary className={SUMMARY_CLASS}>File</summary>
      <div className={POPOVER_CLASS} role="group" aria-label="File commands">
        {fileCommands.map((command) => (
          <button
            key={command.id}
            type="button"
            disabled={!command.enabled}
            data-command-id={command.id}
            title={command.disabledReason ?? undefined}
            className={COMMAND_CLASS}
          >
            {command.label}
          </button>
        ))}
        <p className="mt-2 border-t border-slate-800 pt-2 text-xs leading-relaxed text-amber-200">
          {fileCommands[0]?.disabledReason}
        </p>
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
  const orderedModules = state.order.map((id) =>
    PRIMARY_VIEWS.find((module) => module.id === id),
  ).filter((module): module is (typeof PRIMARY_VIEWS)[number] => module !== undefined);
  return (
    <details className={MENU_CLASS}>
      <summary
        data-command-id={APP_COMMAND_ID.viewManageModules}
        onClick={() => onCommand(APP_COMMAND_ID.viewManageModules)}
        className={SUMMARY_CLASS}
      >
        View
      </summary>
      <div className={`${POPOVER_CLASS} min-w-96`} role="group" aria-label="Workspace modules">
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

export function AppToolstrip({
  moduleState,
  theme,
  shortcutHelpOpen,
  onModuleStateChange,
  onCommand,
  onShortcutHelpOpenChange,
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
        <div className="flex min-w-0 flex-nowrap items-start gap-1 sm:flex-wrap sm:gap-2">
          <FileMenu />
          <ViewMenu state={moduleState} onChange={onModuleStateChange} onCommand={onCommand} />
          <div className="flex min-w-0 max-w-full flex-1 items-stretch gap-1 overflow-x-auto rounded-lg border border-slate-700/80 bg-slate-900/90 p-1">
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
          <details className={MENU_CLASS}>
            <summary className={SUMMARY_CLASS}>Tools</summary>
            <div className={POPOVER_CLASS} role="group" aria-label="Global tools">
              <button
                type="button"
                data-command-id={APP_COMMAND_ID.globalOpenGlossary}
                aria-keyshortcuts={commandShortcut(APP_COMMAND_ID.globalOpenGlossary)}
                aria-label="Open Glossary"
                onClick={() => run(APP_COMMAND_ID.globalOpenGlossary)}
                className={COMMAND_CLASS}
              >
                Open Glossary <span className="float-right text-xs text-slate-500">Alt+G</span>
              </button>
              <button
                type="button"
                data-command-id={APP_COMMAND_ID.globalToggleTheme}
                aria-keyshortcuts={commandShortcut(APP_COMMAND_ID.globalToggleTheme)}
                aria-label={`Toggle Theme; currently ${theme}`}
                onClick={() => run(APP_COMMAND_ID.globalToggleTheme)}
                className={COMMAND_CLASS}
              >
                Theme: {theme === "dark" ? "Dark" : "Light"}
                <span className="float-right text-xs text-slate-500">Alt+T</span>
              </button>
              <button
                ref={shortcutTrigger}
                type="button"
                data-command-id={APP_COMMAND_ID.globalShowShortcuts}
                aria-keyshortcuts={commandShortcut(APP_COMMAND_ID.globalShowShortcuts)}
                aria-label="Keyboard Shortcuts"
                onClick={() => run(APP_COMMAND_ID.globalShowShortcuts)}
                className={COMMAND_CLASS}
              >
                Keyboard Shortcuts <span className="float-right text-xs text-slate-500">?</span>
              </button>
              <button
                type="button"
                data-command-id={APP_COMMAND_ID.globalOpenCurrentModuleHelp}
                aria-keyshortcuts={commandShortcut(APP_COMMAND_ID.globalOpenCurrentModuleHelp)}
                aria-label="Current Module Help"
                onClick={() => run(APP_COMMAND_ID.globalOpenCurrentModuleHelp)}
                className={COMMAND_CLASS}
              >
                Current Module Help <span className="float-right text-xs text-slate-500">F1</span>
              </button>
            </div>
          </details>
        </div>
      </div>
      {shortcutHelpOpen && <ShortcutDialog onClose={closeShortcuts} />}
    </>
  );
}
