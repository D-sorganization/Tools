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
import type { ClubCamera } from "../model/clubCamera";
import {
  loadVisualLayout,
  saveVisualLayout,
} from "../model/visualLayoutPreferences";
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
  readonly clubCamera: ClubCamera;
  readonly setClubCamera: React.Dispatch<React.SetStateAction<ClubCamera>>;
  readonly activatePrimaryView: (view: PrimaryViewId) => void;
  readonly handleCommand: (command: AppCommandId) => void;
}

export function useAppWorkspace(): AppWorkspaceState {
  const [viewState, setViewState] = useState(loadPrimaryViewState);
  const [visualLayout, setVisualLayout] = useState(loadVisualLayout);
  const [theme, setTheme] = usePersistedTheme();
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  useEffect(() => { savePrimaryViewState(viewState); }, [viewState]);
  useEffect(() => { saveVisualLayout(visualLayout); }, [visualLayout]);

  const setModuleHelpOpen = useCallback((moduleHelpOpen: boolean) => {
    setVisualLayout((current) => ({ ...current, moduleHelpOpen }));
  }, []);
  const setClubCamera = useCallback(
    (update: React.SetStateAction<ClubCamera>) => {
      setVisualLayout((current) => ({
        ...current,
        clubCamera: typeof update === "function"
          ? update(current.clubCamera)
          : update,
      }));
    },
    [],
  );

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
  }, [activatePrimaryView, setModuleHelpOpen, setTheme]);

  useGlobalCommandShortcuts(handleCommand);
  return {
    viewState, setViewState, theme, shortcutHelpOpen, setShortcutHelpOpen,
    moduleHelpOpen: visualLayout.moduleHelpOpen, setModuleHelpOpen,
    clubCamera: visualLayout.clubCamera, setClubCamera,
    activatePrimaryView, handleCommand,
  };
}
