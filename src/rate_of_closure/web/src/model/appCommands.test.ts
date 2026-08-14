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
  "file.open_regional_ground_variation_request",
  "file.save_regional_ground_variation_request_as",
  "file.open_regional_ground_execution_job",
  "file.save_regional_ground_execution_job_as",
  "file.save_regional_ground_execution_result_as",
  "file.export_regional_ground_execution_rows_csv",
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
      ({ id }) => id === APP_COMMAND_ID.fileSaveWorkspace,
    );
    expect(disabledCommand).toBeDefined();
    expect(() => requireCommandEnabled(disabledCommand!)).toThrow(
      new CommandUnavailableError(
        APP_COMMAND_ID.fileSaveWorkspace,
        disabledCommand!.disabledReason!,
      ),
    );
  });

  it("enables browser-safe operations and explains unavailable file authority", () => {
    const enabledContextual = new Set<(typeof APP_COMMAND_IDS)[number]>([
      APP_COMMAND_ID.fileOpenRegionalGroundVariationRequest,
      APP_COMMAND_ID.fileSaveRegionalGroundVariationRequestAs,
      APP_COMMAND_ID.fileOpenRegionalGroundExecutionJob,
      APP_COMMAND_ID.fileSaveRegionalGroundExecutionJobAs,
      APP_COMMAND_ID.fileSaveRegionalGroundExecutionResultAs,
      APP_COMMAND_ID.fileExportRegionalGroundExecutionRowsCsv,
    ]);
    expect(
      APP_COMMANDS.filter(({ id }) => enabledContextual.has(id)).every(
        ({ enabled }) => enabled,
      ),
    ).toBe(true);
    const fileCommands = APP_COMMANDS.filter(
      ({ group, id }) => group === "file" && !enabledContextual.has(id),
    );
    expect(fileCommands).toHaveLength(8);
    expect(fileCommands.filter(({ enabled }) => enabled).map(({ id }) => id)).toEqual([
      APP_COMMAND_ID.fileNewWorkspace,
      APP_COMMAND_ID.fileOpenWorkspace,
      APP_COMMAND_ID.fileSaveWorkspaceAs,
      APP_COMMAND_ID.fileImportWorkspace,
      APP_COMMAND_ID.fileExportWorkspace,
      APP_COMMAND_ID.fileCloseWorkspace,
    ]);
    expect(fileCommands.filter(({ enabled }) => !enabled).every(({ disabledReason }) =>
      Boolean(disabledReason?.trim()))).toBe(true);
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
