import type { Ref } from "react";

import type { GeometricVariabilityTs } from "../model/variationGeometry";

interface Props {
  data: GeometricVariabilityTs;
  svgRef?: Ref<SVGSVGElement>;
}

const WIDTH = 640;
const HEIGHT = 190;
const MARGIN = { left: 66, right: 20, top: 20, bottom: 46 };

export function VariationVariabilityTimeline({ data, svgRef }: Props): JSX.Element {
  if (data.sampleTimesS.length === 0) {
    return <p className="text-xs text-slate-500">No valid geometric samples.</p>;
  }
  const displayThreshold = data.criteria.maxValue
    * (data.authorityUnit === "m^3" ? 1e9 : 1e3);
  const finiteValues = data.displayValues.filter(Number.isFinite);
  const maximum = Math.max(
    ...finiteValues,
    displayThreshold,
    1e-6,
  );
  const x = (index: number): number => MARGIN.left + index
    / Math.max(data.sampleTimesS.length - 1, 1)
    * (WIDTH - MARGIN.left - MARGIN.right);
  const y = (radius: number): number => HEIGHT - MARGIN.bottom
    - radius / maximum * (HEIGHT - MARGIN.top - MARGIN.bottom);
  let continuing = false;
  const path = data.displayValues.flatMap((value, index) => {
    if (!Number.isFinite(value)) {
      continuing = false;
      return [];
    }
    const command = `${continuing ? "L" : "M"}${x(index)},${y(value)}`;
    continuing = true;
    return [command];
  }).join(" ");
  return (
    <div className="space-y-1">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full rounded-lg border border-slate-800 bg-slate-950/60"
        role="img"
        aria-label={`${data.metric} and ranked quiet zones over common simulation time`}
      >
        <line x1={MARGIN.left} y1={y(displayThreshold)} x2={WIDTH - MARGIN.right} y2={y(displayThreshold)} stroke="#fbbf24" strokeDasharray="6 4" />
        {data.quietMask.map((quiet, index) => quiet && (
          <rect
            key={index}
            x={x(index)}
            y={MARGIN.top}
            width={Math.max(x(Math.min(index + 1, data.quietMask.length - 1)) - x(index), 2)}
            height={HEIGHT - MARGIN.top - MARGIN.bottom}
            fill="#34d399"
            opacity="0.16"
          />
        ))}
        <path d={path} fill="none" stroke="#38bdf8" strokeWidth="2" />
        <text x={WIDTH / 2} y={HEIGHT - 13} textAnchor="middle" fill="#cbd5e1" fontSize="12">Common Simulation Time [s]</text>
        <text transform={`translate(16 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle" fill="#cbd5e1" fontSize="12">{data.metric} [{data.displayUnit}]</text>
      </svg>
      <p className="text-xs text-slate-500">
        Green bands satisfy the selected threshold; the dashed line is that threshold.
        {` ${data.unavailableCount} samples are unavailable for this metric; adequacy counts and ranked intervals are computed from the loaded ensemble, while selection criteria are retained in the plot definition.`}
      </p>
    </div>
  );
}
