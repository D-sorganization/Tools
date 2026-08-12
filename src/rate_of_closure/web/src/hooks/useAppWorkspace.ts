import { useCallback, useEffect, useLayoutEffect, useState } from "react";

import {
  APP_COMMAND_ID,
  commandForKeyboardEvent,
  type AppCommandId,
} from "../model/appCommands";
import {
  applyAppTheme,
  loadAppTheme,
  saveAppTheme,
  type AppTheme,
} from "../model/appTheme";
import {
  loadPrimaryViewState,
  savePrimaryViewState,
  setPrimaryViewVisibility,
  type PrimaryViewId,
  type PrimaryViewState,
} from "../model/viewPreferences";
import { primaryViewForCommand } from "../model/workspaceCommands";

function usePersistedTheme(): [AppTheme, React.Dispatch<React.SetStateAction<AppTheme>>] {
  const [theme, setTheme] = useState<AppTheme>(loadAppTheme);
  useLayoutEffect(() => applyAppTheme(theme), [theme]);
  useEffect(() => { saveAppTheme(theme); }, [theme]);
  return [theme, setTheme];
}

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (
    target.matches("input, textarea, select") || target.isContentEditable
  );
}

function useGlobalCommandShortcuts(
  handleCommand: (command: AppCommandId) => void,
): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const command = commandForKeyboardEvent(event, isEditableTarget(event.target));
      if (command === null) return;
      event.preventDefault();
      handleCommand(command);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleCommand]);
}

interface AppWorkspaceState {
  readonly viewState: PrimaryViewState;
  readonly setViewState: React.Dispatch<React.SetStateAction<PrimaryViewState>>;
  readonly theme: AppTheme;
  readonly shortcutHelpOpen: boolean;
  readonly setShortcutHelpOpen: (open: boolean) => void;
  readonly moduleHelpOpen: boolean;
  readonly setModuleHelpOpen: (open: boolean) => void;
  readonly activatePrimaryView: (view: PrimaryViewId) => void;
  readonly handleCommand: (command: AppCommandId) => void;
}

export function useAppWorkspace(): AppWorkspaceState {
  const [viewState, setViewState] = useState(loadPrimaryViewState);
  const [theme, setTheme] = usePersistedTheme();
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  const [moduleHelpOpen, setModuleHelpOpen] = useState(false);
  useEffect(() => { savePrimaryViewState(viewState); }, [viewState]);

  const activatePrimaryView = useCallback((active: PrimaryViewId) => {
    setViewState((state) => ({
      ...setPrimaryViewVisibility(state, active, true),
      active,
    }));
  }, []);

  const handleCommand = useCallback((command: AppCommandId) => {
    const destination = primaryViewForCommand(command);
    if (destination !== null) return activatePrimaryView(destination);
    if (command === APP_COMMAND_ID.globalToggleTheme) {
      setTheme((current) => current === "dark" ? "light" : "dark");
    } else if (command === APP_COMMAND_ID.globalShowShortcuts) {
      setShortcutHelpOpen(true);
    } else if (command === APP_COMMAND_ID.globalOpenCurrentModuleHelp) {
      setModuleHelpOpen(true);
    }
  }, [activatePrimaryView, setTheme]);

  useGlobalCommandShortcuts(handleCommand);
  return {
    viewState, setViewState, theme, shortcutHelpOpen, setShortcutHelpOpen,
    moduleHelpOpen, setModuleHelpOpen, activatePrimaryView, handleCommand,
  };
}
