import { describe, expect, it } from "vitest";

import { APP_COMMAND_ID } from "./appCommands";
import { primaryViewForCommand } from "./workspaceCommands";

describe("workspace command routing", () => {
  it("maps direct-view and glossary commands to registered primary modules", () => {
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowImpact)).toBe("explorer");
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowSwing)).toBe("simulation");
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowFlight)).toBe("flight");
    expect(primaryViewForCommand(APP_COMMAND_ID.globalOpenGlossary)).toBe("glossary");
  });

  it("does not invent routes for non-navigation commands", () => {
    expect(primaryViewForCommand(APP_COMMAND_ID.globalToggleTheme)).toBeNull();
    expect(primaryViewForCommand(APP_COMMAND_ID.fileOpenWorkspace)).toBeNull();
  });
});
