/** UI-neutral application command contract shared with Python and automation. */

export const APP_COMMAND_IDS = [
  "file.new_workspace",
  "file.open_workspace",
  "file.open_recent_workspace",
  "file.save_workspace",
  "file.save_workspace_as",
  "file.import_workspace",
  "file.export_workspace",
  "file.close_workspace",
  "view.manage_modules",
  "view.restore_default_workspace",
  "view.show_impact",
  "view.show_swing",
  "view.show_flight",
  "global.open_glossary",
  "global.toggle_theme",
  "global.show_shortcuts",
  "global.open_current_module_help",
] as const;

export type AppCommandId = (typeof APP_COMMAND_IDS)[number];
export type AppCommandGroup = "file" | "view" | "global";

export const APP_COMMAND_ID = Object.freeze({
  fileNewWorkspace: "file.new_workspace",
  fileOpenWorkspace: "file.open_workspace",
  fileOpenRecentWorkspace: "file.open_recent_workspace",
  fileSaveWorkspace: "file.save_workspace",
  fileSaveWorkspaceAs: "file.save_workspace_as",
  fileImportWorkspace: "file.import_workspace",
  fileExportWorkspace: "file.export_workspace",
  fileCloseWorkspace: "file.close_workspace",
  viewManageModules: "view.manage_modules",
  viewRestoreDefaultWorkspace: "view.restore_default_workspace",
  viewShowImpact: "view.show_impact",
  viewShowSwing: "view.show_swing",
  viewShowFlight: "view.show_flight",
  globalOpenGlossary: "global.open_glossary",
  globalToggleTheme: "global.toggle_theme",
  globalShowShortcuts: "global.show_shortcuts",
  globalOpenCurrentModuleHelp: "global.open_current_module_help",
} as const satisfies Record<string, AppCommandId>);

export interface AppCommand {
  readonly id: AppCommandId;
  readonly group: AppCommandGroup;
  readonly label: string;
  readonly enabled: boolean;
  readonly disabledReason: string | null;
  readonly shortcut?: string;
}

const RECENT_DISABLED_REASON =
  "Browsers do not expose persistent recent-file paths without user permission.";
const SAVE_DISABLED_REASON =
  "Browsers cannot overwrite a prior download; use Save Workspace As to download a new copy.";

const enabled = (
  id: AppCommandId,
  group: AppCommandGroup,
  label: string,
  shortcut?: string,
): AppCommand => ({ id, group, label, enabled: true, disabledReason: null, shortcut });

const disabled = (
  id: AppCommandId,
  group: AppCommandGroup,
  label: string,
  disabledReason: string,
): AppCommand => {
  const reason = disabledReason.trim();
  if (reason.length === 0) throw new TypeError("Disabled commands require a reason.");
  return { id, group, label, enabled: false, disabledReason: reason };
};

export const APP_COMMANDS: readonly AppCommand[] = Object.freeze([
  enabled(APP_COMMAND_ID.fileNewWorkspace, "file", "New Workspace"),
  enabled(APP_COMMAND_ID.fileOpenWorkspace, "file", "Open Workspace…"),
  disabled(APP_COMMAND_ID.fileOpenRecentWorkspace, "file", "Open Recent Workspace", RECENT_DISABLED_REASON),
  disabled(APP_COMMAND_ID.fileSaveWorkspace, "file", "Save Workspace", SAVE_DISABLED_REASON),
  enabled(APP_COMMAND_ID.fileSaveWorkspaceAs, "file", "Save Workspace As…"),
  enabled(APP_COMMAND_ID.fileImportWorkspace, "file", "Import View Layout…"),
  enabled(APP_COMMAND_ID.fileExportWorkspace, "file", "Export View Layout…"),
  enabled(APP_COMMAND_ID.fileCloseWorkspace, "file", "Close Workspace"),
  enabled(APP_COMMAND_ID.viewManageModules, "view", "Manage Modules"),
  enabled(APP_COMMAND_ID.viewRestoreDefaultWorkspace, "view", "Restore Default Workspace"),
  enabled(APP_COMMAND_ID.viewShowImpact, "view", "Impact"),
  enabled(APP_COMMAND_ID.viewShowSwing, "view", "Swing"),
  enabled(APP_COMMAND_ID.viewShowFlight, "view", "Flight"),
  enabled(APP_COMMAND_ID.globalOpenGlossary, "global", "Open Glossary", "Alt+G"),
  enabled(APP_COMMAND_ID.globalToggleTheme, "global", "Toggle Theme", "Alt+T"),
  enabled(APP_COMMAND_ID.globalShowShortcuts, "global", "Keyboard Shortcuts", "?"),
  enabled(APP_COMMAND_ID.globalOpenCurrentModuleHelp, "global", "Current Module Help", "F1"),
]);

export class CommandUnavailableError extends Error {
  readonly commandId: AppCommandId;
  readonly reason: string;

  constructor(commandId: AppCommandId, reason: string) {
    super(`${commandId} is unavailable: ${reason}`);
    this.name = "CommandUnavailableError";
    this.commandId = commandId;
    this.reason = reason;
  }
}

export function requireCommandEnabled(command: AppCommand): void {
  if (command.enabled) return;
  if (command.disabledReason === null || command.disabledReason.trim().length === 0) {
    throw new TypeError(`Disabled command ${command.id} requires a reason.`);
  }
  throw new CommandUnavailableError(command.id, command.disabledReason);
}

interface KeyboardCommandInput {
  readonly key: string;
  readonly altKey?: boolean;
  readonly ctrlKey?: boolean;
  readonly metaKey?: boolean;
}

export function commandForKeyboardEvent(
  event: KeyboardCommandInput,
  editableTarget: boolean,
): AppCommandId | null {
  if (editableTarget || event.ctrlKey || event.metaKey) return null;
  const key = event.key.toLowerCase();
  if (event.altKey && key === "g") return APP_COMMAND_ID.globalOpenGlossary;
  if (event.altKey && key === "t") return APP_COMMAND_ID.globalToggleTheme;
  if (!event.altKey && event.key === "?") return APP_COMMAND_ID.globalShowShortcuts;
  if (!event.altKey && key === "f1") return APP_COMMAND_ID.globalOpenCurrentModuleHelp;
  return null;
}

export function commandsInGroup(group: AppCommandGroup): readonly AppCommand[] {
  return APP_COMMANDS.filter((command) => command.group === group);
}
