/** Routing from UI-neutral commands to this client's primary module IDs. */

import { APP_COMMAND_ID, type AppCommandId } from "./appCommands";
import type { PrimaryViewId } from "./viewPreferences";

const PRIMARY_VIEW_BY_COMMAND: Partial<Record<AppCommandId, PrimaryViewId>> = {
  [APP_COMMAND_ID.viewShowImpact]: "explorer",
  [APP_COMMAND_ID.viewShowSwing]: "simulation",
  [APP_COMMAND_ID.viewShowFlight]: "flight",
  [APP_COMMAND_ID.globalOpenGlossary]: "glossary",
};

export function primaryViewForCommand(command: AppCommandId): PrimaryViewId | null {
  return PRIMARY_VIEW_BY_COMMAND[command] ?? null;
}
