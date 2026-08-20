export type LegendPosition = "hidden" | "outside_right" | "inside_top_left" | "inside_top_right";

export interface PlotLayout {
  margin: { left: number; right: number; top: number; bottom: number };
  plotWidth: number;
  plotHeight: number;
  plotRight: number;
  legendX: number;
  legendY: number;
}

const LEFT_MARGIN_PX = 64;
const RIGHT_MARGIN_PX = 20;
const OUTSIDE_LEGEND_RAIL_PX = 190;
const TOP_MARGIN_PX = 42;
const BOTTOM_MARGIN_PX = 46;
const OUTSIDE_LEGEND_INSET_PX = 170;
const INSIDE_RIGHT_LEGEND_INSET_PX = 185;
const LEFT_LEGEND_X_PX = 76;

export function resolvePlotLayout(
  width: number,
  height: number,
  legend: LegendPosition,
): PlotLayout {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new RangeError("plot dimensions must be finite and positive");
  }
  const right = legend === "outside_right" ? OUTSIDE_LEGEND_RAIL_PX : RIGHT_MARGIN_PX;
  const plotWidth = width - LEFT_MARGIN_PX - right;
  const plotHeight = height - TOP_MARGIN_PX - BOTTOM_MARGIN_PX;
  if (plotWidth <= 0 || plotHeight <= 0) {
    throw new RangeError("plot dimensions are too small for the selected legend layout");
  }
  const rightAligned = legend === "outside_right" || legend === "inside_top_right";
  const legendInset = legend === "outside_right"
    ? OUTSIDE_LEGEND_INSET_PX
    : INSIDE_RIGHT_LEGEND_INSET_PX;
  return {
    margin: { left: LEFT_MARGIN_PX, right, top: TOP_MARGIN_PX, bottom: BOTTOM_MARGIN_PX },
    plotWidth,
    plotHeight,
    plotRight: LEFT_MARGIN_PX + plotWidth,
    legendX: rightAligned ? width - legendInset : LEFT_LEGEND_X_PX,
    legendY: legend === "outside_right" ? 52 : 46,
  };
}
