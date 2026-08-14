import { describe, expect, it } from "vitest";

import { APP_COMMAND_ID } from "./appCommands";
import { primaryViewForCommand, viewKindForCommand } from "./workspaceCommands";

describe("workspace command routing", () => {
  it("maps direct-view and glossary commands to registered primary modules", () => {
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowImpact)).toBe("simulation");
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowSwing)).toBe("simulation");
    expect(primaryViewForCommand(APP_COMMAND_ID.viewShowFlight)).toBe("simulation");
    expect(primaryViewForCommand(APP_COMMAND_ID.globalOpenGlossary)).toBe("glossary");
  });

  it("maps direct commands to stable compositor view identities", () => {
    expect(viewKindForCommand(APP_COMMAND_ID.viewShowImpact)).toBe("impact");
    expect(viewKindForCommand(APP_COMMAND_ID.viewShowSwing)).toBe("swing");
    expect(viewKindForCommand(APP_COMMAND_ID.viewShowFlight)).toBe("flight");
    expect(viewKindForCommand(APP_COMMAND_ID.globalOpenGlossary)).toBeNull();
  });

  it("does not invent routes for non-navigation commands", () => {
    expect(primaryViewForCommand(APP_COMMAND_ID.globalToggleTheme)).toBeNull();
    expect(primaryViewForCommand(APP_COMMAND_ID.fileOpenWorkspace)).toBeNull();
  });
});
