import type { CourseLayout } from "../model/course";
import type { FlightPoint } from "../model/flight";
import type {
  FlightSamplePlan, FlightSampleSelection,
} from "../model/flightSampleInspector";
import type { SpatialTargetTs } from "../model/spatialTarget";
import type { TargetRegionTs } from "../model/targets";

export const SIDE_CANVAS_SIZE = { width: 860, height: 260 } as const;
export const TOP_CANVAS_SIZE = { width: 860, height: 220 } as const;

export const responsiveFlightCanvasStyle = (size: { width: number; height: number }) => ({
  width: "100%", height: "auto", aspectRatio: `${size.width} / ${size.height}`,
});

export interface FlightCanvasProps {
  points: FlightPoint[];
  comparisonPoints?: FlightPoint[];
  emptyText?: string;
  layout?: CourseLayout;
  showCourse?: boolean;
  target?: TargetRegionTs;
  spatialTarget?: SpatialTargetTs;
  distanceUnit?: string;
  plan?: FlightSamplePlan | null;
  selection?: FlightSampleSelection | null;
  onSelectionChange?: (selection: FlightSampleSelection | null) => void;
  prominenceRef?: { current: HTMLCanvasElement | null };
}
