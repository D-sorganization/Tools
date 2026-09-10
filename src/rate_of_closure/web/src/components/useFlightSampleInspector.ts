import type { KeyboardEvent, MouseEvent } from "react";

import {
  navigateFlightSamples,
  nearestFlightSample,
  type FlightSamplePlan,
  type FlightSampleSelection,
} from "../model/flightSampleInspector";
import type { FlightPoint } from "../model/flight";
import { spatialTargetHalfExtents, type SpatialTargetTs } from "../model/spatialTarget";

const MARGIN = 34;

interface InspectorOptions {
  plan: FlightSamplePlan | null;
  selection: FlightSampleSelection | null;
  onSelectionChange: (selection: FlightSampleSelection | null) => void;
  points: readonly FlightPoint[];
  comparisonPoints: readonly FlightPoint[];
  spatialTarget?: SpatialTargetTs;
  vertical: "height" | "lateral";
  logicalWidth: number;
  logicalHeight: number;
}

function profileProjection(options: InspectorOptions, cssWidth: number, cssHeight: number) {
  const { plan, points, comparisonPoints, spatialTarget, vertical, logicalWidth, logicalHeight } = options;
  if (!plan || points.length !== plan.rawCount) return [];
  const allPoints = [...points, ...comparisonPoints];
  const center = spatialTarget?.point.appCoordinatesM;
  const extents = spatialTarget ? spatialTargetHalfExtents(spatialTarget) : [0, 0, 0];
  const targetVertical = center
    ? Math.abs(vertical === "height" ? center[1] : center[2]) +
      (vertical === "height" ? extents[1] : extents[2]) : 0;

  // ⚡ Bolt Optimization: Use single-pass loop instead of Math.max(...array.map(...))
  let maxPos0 = center ? center[0] + extents[0] : 0;
  let maxVertical = targetVertical;
  for (let i = 0; i < allPoints.length; i++) {
    const p = allPoints[i].position;
    if (p[0] > maxPos0) maxPos0 = p[0];
    if (vertical === "height") {
      if (p[1] > maxVertical) maxVertical = p[1];
    } else {
      const absP2 = Math.abs(p[2]);
      if (absP2 > maxVertical) maxVertical = absP2;
    }
  }

  const carry = Math.max(10, maxPos0) * 1.05;
  const verticalExtent = vertical === "height"
    ? Math.max(5, maxVertical) * 1.2
    : Math.max(5, maxVertical) * 1.3;
  const zeroY = vertical === "height" ? logicalHeight - MARGIN : logicalHeight / 2;
  const usableY = vertical === "height"
    ? logicalHeight - 2 * MARGIN : logicalHeight / 2 - MARGIN;
  const scale = Math.min((logicalWidth - 2 * MARGIN) / carry, usableY / verticalExtent);
  return plan.samples.map((sample) => {
    const value = vertical === "height" ? sample.heightM : sample.rightM;
    return [
      "current", sample.rawIndex,
      (MARGIN + sample.downrangeM * scale) * cssWidth / logicalWidth,
      (zeroY - value * scale) * cssHeight / logicalHeight,
    ] as const;
  });
}

export function useFlightSampleInspector(options: InspectorOptions) {
  const onClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!options.plan) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const projected = profileProjection(options, rect.width, rect.height);
    const next = nearestFlightSample(
      options.plan, projected, [event.clientX - rect.left, event.clientY - rect.top],
    );
    if (next) options.onSelectionChange(next);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    if (!options.plan) return;
    const commands = {
      ArrowLeft: "previous", ArrowRight: "next", Home: "home", End: "end", Escape: "clear",
    } as const;
    const command = commands[event.key as keyof typeof commands];
    if (!command) return;
    event.preventDefault();
    const rawIndex = navigateFlightSamples(options.plan, options.selection?.rawIndex ?? null, command);
    options.onSelectionChange(rawIndex === null ? null : { cohort: "current", rawIndex });
  };
  return { onClick, onKeyDown, tabIndex: 0 };
}
