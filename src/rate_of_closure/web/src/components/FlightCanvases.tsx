/** Responsive, selectable side and top flight profiles. */

import { useEffect, useRef } from "react";

import { DEFAULT_COURSE_LAYOUT } from "../model/course";
import { observeCanvas } from "./canvasDisplay";
import { drawFlightPanel } from "./flightCanvasDrawing";
import {
  responsiveFlightCanvasStyle,
  SIDE_CANVAS_SIZE,
  TOP_CANVAS_SIZE,
  type FlightCanvasProps,
} from "./flightCanvasContract";
import { spatialTargetSummary } from "./spatialTargetPresentation";
import { useFlightSampleInspector } from "./useFlightSampleInspector";

export function FlightCanvases({
  points,
  comparisonPoints = [],
  emptyText,
  layout,
  showCourse,
  target,
  spatialTarget,
  distanceUnit = "yd",
  plan = null,
  selection = null,
  onSelectionChange = () => undefined,
  prominenceRef,
}: FlightCanvasProps) {
  const sideRef = useRef<HTMLCanvasElement | null>(null);
  const topRef = useRef<HTMLCanvasElement | null>(null);
  const placeholder = emptyText ?? "Run a flight to populate the view.";
  const courseLayout = layout ?? DEFAULT_COURSE_LAYOUT;
  const course = showCourse ?? true;
  const shared = {
    plan,
    selection,
    onSelectionChange,
    points,
    comparisonPoints,
    spatialTarget,
  };
  const sideInspector = useFlightSampleInspector({
    ...shared,
    vertical: "height",
    logicalWidth: SIDE_CANVAS_SIZE.width,
    logicalHeight: SIDE_CANVAS_SIZE.height,
  });
  const topInspector = useFlightSampleInspector({
    ...shared,
    vertical: "lateral",
    logicalWidth: TOP_CANVAS_SIZE.width,
    logicalHeight: TOP_CANVAS_SIZE.height,
  });

  useEffect(() => {
    const drawSide = () => {
      if (!sideRef.current) return;
      drawFlightPanel(
        sideRef.current,
        SIDE_CANVAS_SIZE,
        points,
        comparisonPoints,
        "height",
        placeholder,
        courseLayout,
        course,
        undefined,
        distanceUnit,
        spatialTarget,
        selection?.rawIndex,
      );
    };
    const drawTop = () => {
      if (!topRef.current) return;
      drawFlightPanel(
        topRef.current,
        TOP_CANVAS_SIZE,
        points,
        comparisonPoints,
        "lateral",
        placeholder,
        courseLayout,
        course,
        target,
        distanceUnit,
        spatialTarget,
        selection?.rawIndex,
      );
    };
    const stopSide = observeCanvas(sideRef, drawSide);
    const stopTop = observeCanvas(topRef, drawTop);
    return () => {
      stopSide();
      stopTop();
    };
  }, [
    points,
    comparisonPoints,
    placeholder,
    courseLayout,
    course,
    target,
    spatialTarget,
    distanceUnit,
    selection,
  ]);

  const targetDescription = spatialTarget
    ? ` Plot includes ${spatialTargetSummary(spatialTarget)}`
    : undefined;
  const instructions =
    "Select current primary samples. Arrow keys move; Home/End jump; " +
    "Escape clears.";

  return (
    <div className="grid min-w-0 gap-3">
      <canvas
        {...sideInspector}
        ref={(node) => {
          sideRef.current = node;
          if (prominenceRef) prominenceRef.current = node;
        }}
        width={SIDE_CANVAS_SIZE.width}
        height={SIDE_CANVAS_SIZE.height}
        style={responsiveFlightCanvasStyle(SIDE_CANVAS_SIZE)}
        className="min-h-[180px] w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 sm:min-h-0"
        aria-label="Flight side profile (height vs carry)"
        aria-keyshortcuts="ArrowLeft ArrowRight Home End Escape"
        aria-description={targetDescription}
        title={instructions}
      />
      <canvas
        {...topInspector}
        ref={topRef}
        width={TOP_CANVAS_SIZE.width}
        height={TOP_CANVAS_SIZE.height}
        style={responsiveFlightCanvasStyle(TOP_CANVAS_SIZE)}
        className="min-h-[180px] w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 sm:min-h-0"
        aria-label="Flight top-down view (lateral vs carry)"
        aria-keyshortcuts="ArrowLeft ArrowRight Home End Escape"
        aria-description={targetDescription}
        title={instructions}
      />
    </div>
  );
}
