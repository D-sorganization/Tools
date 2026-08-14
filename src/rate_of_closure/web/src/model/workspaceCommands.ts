/** Routing from UI-neutral commands to this client's primary module IDs. */

import { APP_COMMAND_ID, type AppCommandId } from "./appCommands";
import type { PrimaryViewId } from "./viewPreferences";
import type { ViewKind } from "./viewWorkspace";

const PRIMARY_VIEW_BY_COMMAND: Partial<Record<AppCommandId, PrimaryViewId>> = {
  [APP_COMMAND_ID.viewShowImpact]: "simulation",
  [APP_COMMAND_ID.viewShowSwing]: "simulation",
  [APP_COMMAND_ID.viewShowFlight]: "simulation",
  [APP_COMMAND_ID.globalOpenGlossary]: "glossary",
};

export function primaryViewForCommand(command: AppCommandId): PrimaryViewId | null {
  return PRIMARY_VIEW_BY_COMMAND[command] ?? null;
}

const VIEW_KIND_BY_COMMAND: Partial<Record<AppCommandId, ViewKind>> = {
  [APP_COMMAND_ID.viewShowImpact]: "impact",
  [APP_COMMAND_ID.viewShowSwing]: "swing",
  [APP_COMMAND_ID.viewShowFlight]: "flight",
};

export function viewKindForCommand(command: AppCommandId): ViewKind | null {
  return VIEW_KIND_BY_COMMAND[command] ?? null;
}
