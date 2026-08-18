/**
 * Golf-course scene palette + layout (epic #4125 H7a) — web mirror of
 * `rate_of_closure/ui/course.py` (same blend fractions, same palette
 * anchors), so both UIs render the identical course from the shared
 * chart palette with no widget-level hex.
 */

import { blend, getChartColor } from "./theme";

export interface CourseColors {
  rough: string;
  fairway: string;
  green: string;
  hole: string;
  flag: string;
  tee: string;
}

/** App frame course furniture: x downrange [m]; tee at the origin. */
export interface CourseLayout {
  greenDistanceM: number;
  greenRadiusM: number;
  fairwayHalfWidthM: number;
}

export const DEFAULT_COURSE_LAYOUT: CourseLayout = {
  greenDistanceM: 230.0,
  greenRadiusM: 10.0,
  fairwayHalfWidthM: 16.0,
};

const shade = (color: string, f: number) => blend(color, "#000000", f);
const tint = (color: string, f: number) => blend(color, "#ffffff", f);

/** Derive the course tones from the chart palette (green/red/yellow). */
export function courseColors(): CourseColors {
  const grass = getChartColor(1);
  return {
    rough: shade(grass, 0.45),
    fairway: shade(grass, 0.15),
    green: tint(grass, 0.2),
    hole: shade(grass, 0.85),
    flag: getChartColor(3),
    tee: getChartColor(6),
  };
}
