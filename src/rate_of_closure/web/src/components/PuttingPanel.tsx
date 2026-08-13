/** Putting controls and an SVG green view with phase-coded roll-out. */

import { useLayoutEffect, useMemo, useRef, useState } from "react";

import { DecimalInput } from "./DecimalInput";
import { PuttingVisuals } from "./PuttingVisuals";
import {
  planPuttingSamples, puttingSampleSource, type PuttingSamplePlan,
  puttingContextLabel,
  snapshotPuttingResult,
  validatePuttingResultSummary,
} from "../model/puttingSampleInspector";

import { CLUB_LIBRARY } from "../model/club";
import { FIELD_TO_TERM } from "../model/glossary";
import {
  clubheadSpeedFromBackstroke,
  MINIMAL_PUTTERS,
  type PutterSpec,
  DEFAULT_PUTTER_COR,
  type PuttResult,
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
  /** Production computation authority; injectable for deterministic failure tests. */
  executeStudy?: typeof simulatePutt;
}

interface AcceptedStudy {
  executor: typeof simulatePutt;
  result: PuttResult;
  plan: PuttingSamplePlan;
  context: string;
  holeX: number;
  grade: number;
  aspect: number;
}

export function PuttingPanel({
  onGlossary,
  distanceUnit = "yd",
  executeStudy = simulatePutt,
}: PuttingPanelProps) {
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
  const [selection, setSelection] = useState<{
    accepted: AcceptedStudy; rawIndex: number;
  } | null>(null);
  const acceptedStudy = useRef<AcceptedStudy | null>(null);

  const candidate = useMemo(() => {
    const putter =
      putters.find((p) => p.name === putterName) ?? putters[0];
    try {
      const clubheadSpeed =
        paceMode === "backstroke"
          ? clubheadSpeedFromBackstroke(backstrokeCm / 100)
          : speed;
      const result = snapshotPuttingResult(executeStudy(
        strike(putter, clubheadSpeed),
        { stimpFt: stimp, gradePercent: grade, aspectDeg: aspect },
        distance,
      ));
      const plan = planPuttingSamples(puttingSampleSource(result));
      validatePuttingResultSummary(result, plan);
      const context = puttingContextLabel(
        putter, clubheadSpeed, stimp, grade, aspect, distance,
      );
      const accepted: AcceptedStudy = {
        executor: executeStudy, result, plan, context,
        holeX: distance, grade, aspect,
      };
      return { accepted, error: null };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { accepted: null, error: message.slice(0, 512) };
    }
  }, [
    executeStudy, putters, putterName, paceMode, speed, backstrokeCm,
    stimp, grade, aspect, distance,
  ]);
  const accepted = candidate.accepted ?? (
    acceptedStudy.current?.executor === executeStudy ? acceptedStudy.current : null
  );
  useLayoutEffect(() => {
    if (candidate.accepted !== null) acceptedStudy.current = candidate.accepted;
  }, [candidate]);
  const { error } = candidate;
  const result = accepted?.result ?? null;
  const plan = accepted?.plan ?? null;
  const selectedRawIndex = selection?.accepted === accepted ? selection.rawIndex : null;
  const selectSample = (rawIndex: number | null) => {
    setSelection(rawIndex === null || accepted === null ? null : { accepted, rawIndex });
  };

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
    bounds?: readonly [min: number, max: number],
  ) => (
    <label className="mb-2 flex items-center justify-between gap-2 text-sm">
      <span className="text-slate-300">{label}</span>
      <span className="flex items-center gap-1">
        <DecimalInput
          value={value}
          step={step}
          min={bounds?.[0]}
          max={bounds?.[1]}
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
                [0.2, 6],
              )
            : numberField(
                "Backstroke",
                backstrokeCm,
                setBackstrokeCm,
                1,
                "Backstroke arc length, converted with the simple-pendulum proxy v = A·sqrt(g/L); 10-60 cm typical",
                "cm",
                [5, 100],
              )}
          {numberField(
            "Green speed (stimp)",
            stimp,
            setStimp,
            0.5,
            "Stimpmeter reading; 7 slow - 13 tournament fast (USGA stimpmeter geometry, swing_sim.putting.roll)",
            "ft",
            [3, 16],
          )}
          {numberField(
            "Slope grade",
            grade,
            setGrade,
            0.25,
            "Uniform green slope grade; greens rarely exceed ~5 % (swing_sim.putting.green)",
            "%",
            [0, 10],
          )}
          {numberField(
            "Downhill direction",
            aspect,
            setAspect,
            5,
            "Downhill direction relative to the putt line: 0° ahead, +90° low side left, 180° uphill",
            "°",
            [-360, 360],
          )}
          {numberField(
            "Distance to hole",
            distance,
            setDistance,
            0.1,
            "Ball-to-hole distance along the starting line; 1-15 m typical",
            "m",
            [0.1, 40],
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

      <section aria-label="Green view" className="order-first space-y-4 lg:order-none">
        {error ? (
          <p role="alert" className="rounded border border-red-500/60 bg-red-950/70 px-3 py-2 text-sm text-red-100">
            Attempted putting configuration rejected: {error}. {accepted
              ? "The accepted context below remains displayed."
              : "No accepted putt is available."}
          </p>
        ) : null}
        {accepted ? (
          <p aria-label="Displayed putting result context" className="text-xs text-slate-400">
            Displayed result: {accepted.context}
          </p>
        ) : null}
        <PuttingVisuals result={result} plan={plan} selectedRawIndex={selectedRawIndex}
          onSelectionChange={selectSample} holeX={accepted?.holeX ?? distance}
          grade={accepted?.grade ?? grade} aspect={accepted?.aspect ?? aspect} />
      </section>
    </div>
  );
}
