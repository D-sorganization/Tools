/** Defensive theme persistence and DOM application. */

export type AppTheme = "dark" | "light";

export const APP_THEME_STORAGE_KEY = "rate-of-closure.web.theme.v1";

interface ThemeReader {
  getItem(key: string): string | null;
}

interface ThemeWriter {
  setItem(key: string, value: string): void;
}

export function loadAppTheme(storage?: ThemeReader | null): AppTheme {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    return target?.getItem(APP_THEME_STORAGE_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function saveAppTheme(
  theme: AppTheme,
  storage?: ThemeWriter | null,
): boolean {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    if (target === null) return false;
    target.setItem(APP_THEME_STORAGE_KEY, theme);
    return true;
  } catch {
    return false;
  }
}

export function applyAppTheme(
  theme: AppTheme,
  root: HTMLElement = document.documentElement,
): void {
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
}
