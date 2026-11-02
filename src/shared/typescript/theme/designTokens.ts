import designTokens from '../../design_tokens.json';
import themesData from '../../theme-definitions/themes.json';

export interface ThemeColors {
  bg: string;
  group_bg: string;
  border: string;
  text: string;
  text_secondary: string;
  label: string;
  focus: string;
  input_bg: string;
  accent: string;
  title_bg: string;
  title_border: string;
  table_header: string;
  table_alt: string;
  button_hover: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  link: string;
  link_hover: string;
  selection_bg: string;
  selection_text: string;
  [key: string]: string;
}

export interface DesignTokens {
  themes: Record<string, ThemeColors>;
  spacing: Record<string, string>;
  typography: Record<string, string>;
  radii: Record<string, string>;
}

// themes.json is the single source of theme colors (Tools #5917); the themes
// table in design_tokens.json is a generated copy (scripts/sync_design_tokens.py).
const catalogThemes: Record<string, ThemeColors> = {};
for (const [id, theme] of Object.entries(themesData.themes)) {
  catalogThemes[id] = { ...theme.colors, ...theme.semantic } as ThemeColors;
}

const tokens: DesignTokens = { ...(designTokens as any), themes: catalogThemes };

/**
 * Generates a string of CSS custom properties for a given theme.
 * Example output:
 * --color-bg: #ffffff;
 * --spacing-md: 16px;
 */
export function generateCssCustomProperties(themeName: string = 'light'): string {
  const theme = tokens.themes[themeName];
  if (!theme) {
    throw new Error(`Theme '${themeName}' not found in design tokens.`);
  }

  const lines: string[] = [];

  // Colors
  Object.entries(theme).forEach(([key, value]) => {
    lines.push(`--color-${key.replace(/_/g, '-')}: ${value};`);
  });

  // Spacing
  Object.entries(tokens.spacing || {}).forEach(([key, value]) => {
    lines.push(`--spacing-${key}: ${value};`);
  });

  // Radii
  Object.entries(tokens.radii || {}).forEach(([key, value]) => {
    lines.push(`--radius-${key}: ${value};`);
  });

  // Typography
  Object.entries(tokens.typography || {}).forEach(([key, value]) => {
    lines.push(`--font-${key.replace(/_/g, '-')}: ${value};`);
  });

  return lines.join('\n');
}

/**
 * Applies CSS custom properties to a DOM element (usually document.documentElement).
 */
export function applyThemeToElement(element: HTMLElement, themeName: string = 'light'): void {
  const theme = tokens.themes[themeName];
  if (!theme) return;

  Object.entries(theme).forEach(([key, value]) => {
    element.style.setProperty(`--color-${key.replace(/_/g, '-')}`, value);
  });

  Object.entries(tokens.spacing || {}).forEach(([key, value]) => {
    element.style.setProperty(`--spacing-${key}`, value);
  });

  Object.entries(tokens.radii || {}).forEach(([key, value]) => {
    element.style.setProperty(`--radius-${key}`, value);
  });

  Object.entries(tokens.typography || {}).forEach(([key, value]) => {
    element.style.setProperty(`--font-${key.replace(/_/g, '-')}`, value);
  });
}

export default tokens;
