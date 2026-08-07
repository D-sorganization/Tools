import { describe, expect, it } from "vitest";

import {
  APP_COMMANDS,
  APP_COMMAND_ID,
  APP_COMMAND_IDS,
  CommandUnavailableError,
  commandForKeyboardEvent,
  requireCommandEnabled,
} from "./appCommands";

const CANONICAL_COMMAND_IDS = [
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

describe("application command registry", () => {
  it("matches the complete ordered UI-neutral Python command contract", () => {
    expect(APP_COMMAND_IDS).toEqual(CANONICAL_COMMAND_IDS);
    expect(APP_COMMANDS.map(({ id }) => id)).toEqual(CANONICAL_COMMAND_IDS);
    expect(new Set(APP_COMMAND_IDS).size).toBe(APP_COMMAND_IDS.length);
  });

  it("enforces enabled/disabled reason invariants", () => {
    expect(APP_COMMANDS.every(({ enabled, disabledReason }) =>
      enabled ? disabledReason === null : Boolean(disabledReason?.trim()),
    )).toBe(true);
    const disabledCommand = APP_COMMANDS.find(
      ({ id }) => id === APP_COMMAND_ID.fileNewWorkspace,
    );
    expect(disabledCommand).toBeDefined();
    expect(() => requireCommandEnabled(disabledCommand!)).toThrow(
      new CommandUnavailableError(
        APP_COMMAND_ID.fileNewWorkspace,
        disabledCommand!.disabledReason!,
      ),
    );
  });

  it("keeps unavailable workspace-document commands truthfully disabled", () => {
    const fileCommands = APP_COMMANDS.filter(({ group }) => group === "file");
    expect(fileCommands).toHaveLength(8);
    expect(fileCommands.every(({ enabled }) => !enabled)).toBe(true);
    expect(fileCommands.every(({ disabledReason }) =>
      disabledReason?.includes("workspace document adapter"))).toBe(true);
  });

  it("resolves global shortcuts without stealing editable-field keystrokes", () => {
    expect(commandForKeyboardEvent({ key: "g", altKey: true }, false))
      .toBe(APP_COMMAND_ID.globalOpenGlossary);
    expect(commandForKeyboardEvent({ key: "?" }, false))
      .toBe(APP_COMMAND_ID.globalShowShortcuts);
    expect(commandForKeyboardEvent({ key: "F1" }, false))
      .toBe(APP_COMMAND_ID.globalOpenCurrentModuleHelp);
    expect(commandForKeyboardEvent({ key: "g", altKey: true }, true)).toBeNull();
  });
});
