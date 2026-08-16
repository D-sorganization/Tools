/** Defensive publication boundary for computed plot data. */
import {
  MAX_ABS_PLOT_VALUE,
  MAX_PLOT_SAMPLES,
  MAX_PLOT_SERIES,
  MAX_PLOT_VERTICES,
} from "./plotPointInspector";
import { type PlotData } from "./plotspec";

export function snapshotPlotData(data: PlotData): PlotData {
  if (
    data.x.length < 1 ||
    data.x.length > MAX_PLOT_SAMPLES ||
    data.x.some(
      (value) =>
        !Number.isFinite(value) || Math.abs(value) > MAX_ABS_PLOT_VALUE,
    )
  ) {
    throw new RangeError(
      `plot evidence must contain 1..${MAX_PLOT_SAMPLES} finite bounded samples`,
    );
  }
  if (
    data.series.length > MAX_PLOT_SERIES ||
    data.x.length * data.series.length > MAX_PLOT_VERTICES
  ) {
    throw new RangeError("plot evidence exceeds the bounded series/vertex contract");
  }
  const x = Object.freeze([...data.x]);
  const series = Object.freeze(
    data.series.map((item) => {
      if (
        item.values.length !== x.length ||
        typeof item.label !== "string" ||
        item.label.length < 1 ||
        item.label.length > 512 ||
        item.values.some(
          (value) =>
            typeof value !== "number" ||
            value === Infinity ||
            value === -Infinity ||
            (Number.isFinite(value) && Math.abs(value) > MAX_ABS_PLOT_VALUE),
        )
      ) {
        throw new RangeError(
          "plot series must align and contain bounded values or NaN gaps",
        );
      }
      return Object.freeze({
        label: item.label,
        values: Object.freeze([...item.values]),
      });
    }),
  );
  const spec = Object.freeze({
    ...data.spec,
    y_keys: Object.freeze([...data.spec.y_keys]),
  });
  return Object.freeze({
    spec,
    x,
    series,
    xLabel: data.xLabel,
    yLabel: data.yLabel,
  });
}
