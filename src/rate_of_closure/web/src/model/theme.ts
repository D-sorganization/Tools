/**
 * Shared chart palette for canvas drawing (epic #4125 H7a).
 *
 * Mirrors the UpstreamDrift theme palette (shared/theme-definitions/
 * themes.json `chartColors`) so canvas widgets derive scene colors
 * from one palette instead of hard-coding hex per widget — the web
 * twin of `shared.python.theme.matplotlib_style.get_chart_color`.
 */

export const CHART_COLORS: readonly string[] = [
  "#0A84FF", // blue
  "#30D158", // green
  "#FF9F0A", // orange
  "#FF375F", // red
  "#BF5AF2", // purple
  "#64D2FF", // cyan
  "#FFD60A", // yellow
  "#AC8E68", // brown
];

export function getChartColor(index: number): string {
  return CHART_COLORS[((index % CHART_COLORS.length) + CHART_COLORS.length) % CHART_COLORS.length];
}

/** Linear RGB blend of two #rrggbb colors, `fraction` toward `other`. */
export function blend(color: string, other: string, fraction: number): string {
  const parse = (hex: string): [number, number, number] | null => {
    const value = hex.trim().replace(/^#/, "");
    if (!/^[0-9a-fA-F]{6}$/.test(value)) return null;
    return [
      parseInt(value.slice(0, 2), 16),
      parseInt(value.slice(2, 4), 16),
      parseInt(value.slice(4, 6), 16),
    ];
  };
  const a = parse(color);
  const b = parse(other);
  if (!a || !b) return color;
  const t = Math.min(Math.max(fraction, 0), 1);
  const mix = a.map((ca, i) => Math.round(ca + (b[i] - ca) * t));
  return `#${mix.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/** Append an alpha channel to a #rrggbb color for canvas fills. */
export function withAlpha(color: string, alpha: number): string {
  const a = Math.round(Math.min(Math.max(alpha, 0), 1) * 255);
  return `${color}${a.toString(16).padStart(2, "0")}`;
}
