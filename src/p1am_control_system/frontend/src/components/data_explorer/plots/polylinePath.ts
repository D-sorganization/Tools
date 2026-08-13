export type PlotPoint = readonly [number, number];

/** Build a segmented SVG polyline path, treating non-finite samples as gaps. */
export function buildPolylinePath(
  points: readonly PlotPoint[],
  px: (value: number) => number,
  py: (value: number) => number,
): string {
  let path = "";
  let penDown = false;
  for (const [dataX, dataY] of points) {
    if (!Number.isFinite(dataX) || !Number.isFinite(dataY)) {
      penDown = false;
      continue;
    }
    if (path.length > 0) path += " ";
    path += `${penDown ? "L" : "M"}${px(dataX)},${py(dataY)}`;
    penDown = true;
  }
  return path;
}
