/** Putting controls and an SVG green view with phase-coded roll-out. */

import { useMemo, useState } from "react";

import { DecimalInput } from "./DecimalInput";

import { CLUB_LIBRARY } from "../model/club";
import { FIELD_TO_TERM } from "../model/glossary";
import {
  captureSpeedMps,
  clubheadSpeedFromBackstroke,
  MINIMAL_PUTTERS,
  type PutterSpec,
  DEFAULT_PUTTER_COR,
  simulatePutt,
  strike,
} from "../model/putting";
import { formatDistanceM } from "../model/units";

/** Library putters first (H1 reconciliation), minimal specs fallback. */
function putterChoices(): PutterSpec[] {
  const library = CLUB_LIBRARY.filter((c) => c.clubType === "Putter").map(
    (c) => ({
      name: c.name,
      headMassKg: c.headMassKg,
      loftDeg: c.loftDeg,
      cor: DEFAULT_PUTTER_COR,
    }),
  );
  return library.length > 0 ? library : MINIMAL_PUTTERS;
}

const ROWS: { key: string; label: string; explanation: string }[] = [
  {
    key: "puttRolloutM",
    label: "Roll-Out Distance",
    explanation:
      "How far the ball travels before stopping (or dropping). The skid " +
      "phase sheds speed at the sliding-friction rate, then pure roll " +
      "decelerates at the stimp-derived rolling rate — faster greens mean " +
      "a lower rolling coefficient and a longer roll-out for the same pace.",
  },
  {
    key: "puttSkidM",
    label: "Skid Distance",
    explanation:
      "Ground covered while the ball is still sliding rather than rolling. " +
      "A struck putt leaves the face with backspin, so friction must first " +
      "spin it up to pure roll; the transition happens where ball speed " +
      "equals surface spin speed (v = ωr).",
  },
  {
    key: "puttSkidPct",
    label: "Skid Share of Putt",
    explanation:
      "The skid distance as a share of the whole putt. Good strokes keep " +
      "this small — the classic no-spin result is pure roll at 5/7 of " +
      "launch speed, and more backspin extends the skid.",
  },
  {
    key: "puttTimeS",
    label: "Time To Rest",
    explanation:
      "Elapsed time from impact until the ball stops or drops. Rolling " +
      "deceleration is constant on a uniform green, so time grows linearly " +
      "with the speed the roll phase starts at.",
  },
  {
    key: "puttBreakM",
    label: "Break",
    explanation:
      "Lateral drift of the ball off the starting line (positive = left), " +
      "caused by the in-plane component of gravity on the sloped green. " +
      "Break grows fastest late in the putt, when the ball is slow.",
  },
  {
    key: "puttSpeedAtHoleMps",
    label: "Speed At The Hole",
    explanation:
      "Ball speed when it first crosses the hole mouth. The putt drops " +
      "only if this is at or below the geometric capture bound — the ball " +
      "must fall half its diameter while crossing the opening.",
  },
  {
    key: "puttMargin",
    label: "Holed / Miss Margin",
    explanation:
      "Holed putts: how far under the capture-speed bound the ball crossed " +
      "the hole. Missed putts: the distance from the ball's resting place " +
      "back to the hole — the length of the comebacker.",
  },
];

/** Single distance-format chokepoint — follows the session distance
 * display unit (#4125 H6: yards default, metres option). */
function formatDistance(value: number, unit: string): string {
  return formatDistanceM(value, unit, 2);
}

interface PuttingPanelProps {
  onGlossary?: (term: string) => void;
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
}

export function PuttingPanel({ onGlossary, distanceUnit = "yd" }: PuttingPanelProps) {
  const formatM = (value: number) => formatDistance(value, distanceUnit);
  const putters = useMemo(putterChoices, []);
  const [putterName, setPutterName] = useState(putters[0].name);
  const [paceMode, setPaceMode] = useState<"speed" | "backstroke">("speed");
  const [speed, setSpeed] = useState(1.8);
  const [backstrokeCm, setBackstrokeCm] = useState(30);
  const [stimp, setStimp] = useState(10);
  const [grade, setGrade] = useState(0);
  const [aspect, setAspect] = useState(90);
  const [distance, setDistance] = useState(3);
  const [explained, setExplained] = useState(ROWS[0].key);

  const result = useMemo(() => {
    const putter =
      putters.find((p) => p.name === putterName) ?? putters[0];
    try {
      const clubheadSpeed =
        paceMode === "backstroke"
          ? clubheadSpeedFromBackstroke(backstrokeCm / 100)
          : speed;
      return simulatePutt(
        strike(putter, clubheadSpeed),
        { stimpFt: stimp, gradePercent: grade, aspectDeg: aspect },
        distance,
      );
    } catch {
      return null;
    }
  }, [putters, putterName, paceMode, speed, backstrokeCm, stimp, grade, aspect, distance]);

  const values: Record<string, string> = result
    ? {
        puttRolloutM: formatM(result.totalDistanceM),
        puttSkidM: formatM(result.skidDistanceM),
        puttSkidPct: `${(
          (100 * result.skidDistanceM) /
          Math.max(result.totalDistanceM, 1e-9)
        ).toFixed(1)} %`,
        puttTimeS: `${result.timeS.toFixed(2)} s`,
        puttBreakM: formatM(result.breakM),
        puttSpeedAtHoleMps:
          result.speedAtHoleMps !== null
            ? `${result.speedAtHoleMps.toFixed(2)} m/s`
            : "— (never reached)",
        puttMargin: result.holed
          ? `HOLED (+${(result.marginMps ?? 0).toFixed(2)} m/s under bound)`
          : `miss by ${formatM(result.missDistanceM ?? 0)}`,
      }
    : {};
  const explainedRow = ROWS.find((r) => r.key === explained) ?? ROWS[0];

  const numberField = (
    label: string,
    value: number,
    set: (v: number) => void,
    step: number,
    title: string,
    suffix: string,
  ) => (
    <label className="mb-2 flex items-center justify-between gap-2 text-sm">
      <span className="text-slate-300">{label}</span>
      <span className="flex items-center gap-1">
        <DecimalInput
          value={value}
          step={step}
          aria-label={`${label} ${suffix}`.trim()}
          title={title}
          onCommit={set}
          className="w-24 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-right text-slate-100 focus:border-blue-500 focus:outline-none"
        />
        <span className="text-slate-400">{suffix}</span>
      </span>
    </label>
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <section aria-label="Putt setup" className="space-y-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Putt Setup
          </h2>
          <label className="mb-2 flex items-center justify-between gap-2 text-sm">
            <span className="text-slate-300">Putter</span>
            <select
              value={putterName}
              title="Putter head used for the impact model (library putters when available); head mass and loft drive ball speed and launch spin"
              onChange={(e) => setPutterName(e.target.value)}
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
            >
              {putters.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label className="mb-2 flex items-center justify-between gap-2 text-sm">
            <span className="text-slate-300">Pace input</span>
            <select
              value={paceMode}
              title="Set the stroke pace directly as clubhead speed, or as a pendulum backstroke length (v = A·sqrt(g/L))"
              onChange={(e) =>
                setPaceMode(e.target.value as "speed" | "backstroke")
              }
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
            >
              <option value="speed">Clubhead speed</option>
              <option value="backstroke">Backstroke length</option>
            </select>
          </label>
          {paceMode === "speed"
            ? numberField(
                "Clubhead speed",
                speed,
                setSpeed,
                0.05,
                "Clubhead speed at impact; 0.5-3 m/s covers putts inside 15 m (swing_sim.putting.impact)",
                "m/s",
              )
            : numberField(
                "Backstroke",
                backstrokeCm,
                setBackstrokeCm,
                1,
                "Backstroke arc length, converted with the simple-pendulum proxy v = A·sqrt(g/L); 10-60 cm typical",
                "cm",
              )}
          {numberField(
            "Green speed (stimp)",
            stimp,
            setStimp,
            0.5,
            "Stimpmeter reading; 7 slow - 13 tournament fast (USGA stimpmeter geometry, swing_sim.putting.roll)",
            "ft",
          )}
          {numberField(
            "Slope grade",
            grade,
            setGrade,
            0.25,
            "Uniform green slope grade; greens rarely exceed ~5 % (swing_sim.putting.green)",
            "%",
          )}
          {numberField(
            "Downhill direction",
            aspect,
            setAspect,
            5,
            "Downhill direction relative to the putt line: 0° ahead, +90° low side left, 180° uphill",
            "°",
          )}
          {numberField(
            "Distance to hole",
            distance,
            setDistance,
            0.1,
            "Ball-to-hole distance along the starting line; 1-15 m typical",
            "m",
          )}
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Putt Results
          </h2>
          {ROWS.map((row) => (
            <button
              key={row.key}
              type="button"
              onClick={() => setExplained(row.key)}
              aria-pressed={explained === row.key}
              title={`Click for a plain-language explanation of ${row.label}`}
              className={
                "mb-1 flex w-full items-center justify-between rounded-lg border px-3 py-1.5 text-sm transition-all " +
                (explained === row.key
                  ? "border-sky-400/60 bg-sky-500/10 ring-1 ring-sky-400/40"
                  : "border-slate-800 bg-slate-900/40 hover:border-slate-600")
              }
            >
              <span className="text-slate-300">{row.label}</span>
              <span className="font-semibold text-slate-100">
                {values[row.key] ?? "—"}
              </span>
            </button>
          ))}
        </div>

        <div
          aria-label="Explanation"
          className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 text-sm shadow-lg shadow-black/20 backdrop-blur"
        >
          <h3 className="mb-1 font-semibold text-slate-200">
            {explainedRow.label}
          </h3>
          <p className="text-slate-400">{explainedRow.explanation}</p>
          <button
            type="button"
            title="Open the Glossary at the matching term"
            onClick={() => onGlossary?.(FIELD_TO_TERM[explainedRow.key] ?? "")}
            className="mt-2 text-sky-400 hover:text-sky-300"
          >
            Glossary
          </button>
        </div>
      </section>

      <section aria-label="Green view" className="space-y-4">
        <GreenView result={result} holeX={distance} grade={grade} aspect={aspect} />
        <SpeedPlot result={result} />
      </section>
    </div>
  );
}

function GreenView({
  result,
  holeX,
  grade,
  aspect,
}: {
  result: ReturnType<typeof simulatePutt> | null;
  holeX: number;
  grade: number;
  aspect: number;
}) {
  const width = 640;
  const height = 320;
  if (!result) {
    return <p className="text-sm text-slate-400">Inputs out of range.</p>;
  }
  const xs = result.pathXM;
  const ys = result.pathYM;
  const maxX = Math.max(holeX + 0.5, ...xs) + 0.3;
  const minX = Math.min(0, ...xs) - 0.3;
  const spanY = Math.max(0.8, 2 * Math.max(...ys.map(Math.abs), 0.3));
  const sx = (x: number) => ((x - minX) / (maxX - minX)) * width;
  const sy = (y: number) => height / 2 - (y / spanY) * height;
  const toPoints = (from: number, to: number) =>
    xs
      .slice(from, to)
      .map((x, i) => `${sx(x).toFixed(1)},${sy(ys[from + i]).toFixed(1)}`)
      .join(" ");
  const split = result.skidEndIndex;
  const arrowLen = 40;
  const ax = sx(holeX * 0.5);
  const ay = sy(0);
  return (
    <figure
      aria-label="Top-down green view: skid phase orange, pure roll green, hole circle, downhill arrow"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur"
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Putt path on the green"
        className="w-full rounded-lg bg-emerald-950/60"
      >
        <polyline
          points={toPoints(0, split + 1)}
          fill="none"
          stroke="#fb923c"
          strokeWidth={3}
        />
        <polyline
          points={toPoints(split, xs.length)}
          fill="none"
          stroke="#4ade80"
          strokeWidth={3}
        />
        <circle
          cx={sx(holeX)}
          cy={sy(0)}
          r={Math.max(5, (0.054 / (maxX - minX)) * width)}
          fill="none"
          stroke="#f8fafc"
          strokeWidth={2}
        />
        <circle cx={sx(0)} cy={sy(0)} r={4} fill="#f8fafc" />
        {grade > 0 && (
          <g stroke="#94a3b8" strokeWidth={2}>
            <line
              x1={ax}
              y1={ay}
              x2={ax + arrowLen * Math.cos((aspect * Math.PI) / 180)}
              y2={ay - arrowLen * Math.sin((aspect * Math.PI) / 180)}
              markerEnd="url(#downhill-arrow)"
            />
            <defs>
              <marker
                id="downhill-arrow"
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L6,3 L0,6 z" fill="#94a3b8" />
              </marker>
            </defs>
          </g>
        )}
        {result.holed && (
          <text
            x={sx(holeX)}
            y={sy(0) - 14}
            textAnchor="middle"
            fill="#4ade80"
            fontSize="13"
          >
            HOLED
          </text>
        )}
      </svg>
      <figcaption className="mt-2 text-xs text-slate-400">
        Orange = skid phase, green = pure roll; the circle is the hole, the
        grey arrow points downhill. Left in the plot is the putt's left (+y).
      </figcaption>
    </figure>
  );
}

function SpeedPlot({
  result,
}: {
  result: ReturnType<typeof simulatePutt> | null;
}) {
  const width = 640;
  const height = 180;
  if (!result) return null;
  const distances: number[] = [0];
  for (let i = 1; i < result.pathXM.length; i++) {
    distances.push(
      distances[i - 1] +
        Math.hypot(
          result.pathXM[i] - result.pathXM[i - 1],
          result.pathYM[i] - result.pathYM[i - 1],
        ),
    );
  }
  const maxD = Math.max(distances[distances.length - 1], 0.1);
  const maxV = Math.max(...result.speedsMps, captureSpeedMps()) * 1.08;
  const sx = (d: number) => (d / maxD) * (width - 20) + 10;
  const sy = (v: number) => height - 16 - (v / maxV) * (height - 32);
  const points = distances
    .map((d, i) => `${sx(d).toFixed(1)},${sy(result.speedsMps[i]).toFixed(1)}`)
    .join(" ");
  const splitD = distances[Math.min(result.skidEndIndex, distances.length - 1)];
  return (
    <figure
      aria-label="Ball speed versus distance with the capture-speed bound and the skid-to-roll transition marked"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur"
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Speed vs distance plot"
        className="w-full rounded-lg bg-slate-950/60"
      >
        <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth={2.5} />
        <line
          x1={10}
          x2={width - 10}
          y1={sy(captureSpeedMps())}
          y2={sy(captureSpeedMps())}
          stroke="#f87171"
          strokeWidth={1.5}
          strokeDasharray="6 4"
        />
        <line
          x1={sx(splitD)}
          x2={sx(splitD)}
          y1={12}
          y2={height - 16}
          stroke="#fb923c"
          strokeWidth={1.5}
          strokeDasharray="3 4"
        />
      </svg>
      <figcaption className="mt-2 text-xs text-slate-400">
        Blue: ball speed vs distance rolled. Red dashes: the capture-speed
        bound the ball must be under at the hole. Orange dashes: skid → pure
        roll transition.
      </figcaption>
    </figure>
  );
}
