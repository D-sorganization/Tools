import {
  GOLF_BALL_RADIUS_M,
  type BallSetup,
} from "../model/ballSetup";

interface Props {
  setup: BallSetup;
}

/** Side-elevation inspection diagram with the same physical reference as contact geometry. */
export function BallSetupDiagram({ setup }: Props) {
  const centerY = GOLF_BALL_RADIUS_M + setup.teeHeightM;
  const scale = 900;
  const groundY = 112;
  const centerPx = groundY - centerY * scale;
  const radiusPx = GOLF_BALL_RADIUS_M * scale;
  const ballBottom = centerPx + radiusPx;
  return (
    <figure className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
      <svg
        viewBox="0 0 180 130"
        role="img"
        aria-label={`${setup.supportMode === "tee" ? "Tee" : "Ground"} ball support side elevation`}
        className="mx-auto h-32 w-full max-w-xs"
      >
        <defs>
          <radialGradient id="ball-shading" cx="35%" cy="30%">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="0.7" stopColor="#e2e8f0" />
            <stop offset="1" stopColor="#94a3b8" />
          </radialGradient>
          <linearGradient id="tee-shading" x1="0" x2="1">
            <stop offset="0" stopColor="#7f1d1d" />
            <stop offset="0.5" stopColor="#ef4444" />
            <stop offset="1" stopColor="#7f1d1d" />
          </linearGradient>
        </defs>
        <rect x="0" y={groundY} width="180" height="18" fill="#123524" />
        <line x1="0" y1={groundY} x2="180" y2={groundY} stroke="#4ade80" />
        {setup.supportMode === "tee" && (
          <g aria-label="Representative Tee">
            <path
              d={`M 87 ${groundY} L 89 ${ballBottom + 3} L 91 ${ballBottom + 3} L 93 ${groundY} Z`}
              fill="url(#tee-shading)"
            />
            <path
              d={`M 80 ${ballBottom + 3} Q 90 ${ballBottom + 10} 100 ${ballBottom + 3}`}
              fill="none"
              stroke="#ef4444"
              strokeWidth="3"
              strokeLinecap="round"
            />
          </g>
        )}
        <circle cx="90" cy={centerPx} r={radiusPx} fill="url(#ball-shading)" stroke="#cbd5e1" />
        <text x="6" y="14" fill="#94a3b8" fontSize="9">
          {setup.supportMode === "tee"
            ? `Bottom Clearance: ${(setup.teeHeightM * 1000).toFixed(1)} mm`
            : "Bottom Clearance: 0 mm"}
        </text>
      </svg>
      <figcaption className="text-center text-xs text-slate-500">
        Ground Plane To Ball Bottom — Side Elevation
      </figcaption>
    </figure>
  );
}
