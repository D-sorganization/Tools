/**
 * Putting tab — stroke, putter, green controls and the 2-D green view.
 *
 * React parity for the shared putting model (#4800 P1-P5, P7). One
 * chokepoint runs the putt: `evaluatePuttWithTrajectory` performs P1's
 * impact solve, P2's surface integration and P5's `putting_result/2`
 * record in a single call, and the tab presents that record — never a
 * second, differently derived set of numbers.
 *
 * Playback (#4800 P8) is wiring, not new machinery: the retained
 * integrator samples are lifted by `puttPlaybackSamples`, interpolated by
 * the shared structural `PlaybackTimeline`, and driven by the shared
 * `PlaybackTransportBar` — the same three modules the ball-flight surface
 * uses. Nothing is re-simulated during playback and no second transport
 * exists.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { PlaybackTransportBar } from "./PlaybackTransportBar";
import { PuttingControls, type ImportedGreenState } from "./PuttingControls";
import { DEFAULT_PUTT_SETUP, type PuttSetup } from "./puttingSetup";
import { PuttingVisuals } from "./PuttingVisuals";
import {
  PUTTING_RESULT_ROWS,
  puttingResultValues,
} from "./puttingResultRows";
import {
  planPuttingSamples,
  puttingContextLabel,
  puttingSampleSource,
  snapshotPuttingResult,
  validatePuttingResultSummary,
  type PuttingSamplePlan,
} from "../model/puttingSampleInspector";

import { CLUB_LIBRARY } from "../model/club";
import { GLOSSARY } from "../model/glossary";
import {
  headMoiForStrike,
  putterHeadFromLibrary,
  putterSpec,
  twistResponse,
  type PutterHeadDocument,
  type PutterTwist,
} from "../model/putterHead";
import {
  clubheadSpeedFromBackstroke,
  MINIMAL_PUTTERS,
  type PuttResult,
} from "../model/putting";
import { PlaybackTimeline, type PlaybackFrame } from "../model/flightPlayback";
import { puttPlaybackSamples } from "../model/puttPlayback";
import { planarSurface, type GreenSurface } from "../model/puttingGreen";
import {
  PUTTING_RESULT_KERNEL,
  type PuttingResultDocument,
} from "../model/puttingResultWire";
import {
  evaluatePuttWithTrajectory,
  puttStroke,
  type PuttScenario,
} from "../model/puttingScenario";
import { formatDistanceM } from "../model/units";

/** Library putters first (H1 reconciliation), minimal specs fallback. */
function putterChoices(): PutterHeadDocument[] {
  const library = CLUB_LIBRARY.filter((club) => club.clubType === "Putter").map(
    (club) =>
      putterHeadFromLibrary(club.name, {
        headMassKg: club.headMassKg,
        loftDeg: club.loftDeg,
      }),
  );
  if (library.length > 0) return library;
  return MINIMAL_PUTTERS.map((spec) => ({
    name: spec.name,
    head_mass_kg: spec.headMassKg,
    loft_deg: spec.loftDeg,
    cor: spec.cor,
    provenance: { source_kind: "library" as const, library_name: spec.name },
  }));
}

/** Single distance-format chokepoint — follows the session unit. */
function formatDistance(value: number, unit: string): string {
  return formatDistanceM(value, unit, 2);
}

/**
 * Build the fully specified scenario one accepted putt is run from.
 *
 * `surface` is supplied by the caller rather than derived here: the
 * planar grade/aspect green from `setup`, or a heightfield imported
 * through `greenSurfaceFromDocument` (ADR-0045 F2's "Import green…"
 * action) that replaces it outright — the same authority rule the Qt
 * `PuttingGreenControls.surface()` follows.
 */
function puttScenario(
  head: PutterHeadDocument,
  clubheadSpeedMps: number,
  setup: PuttSetup,
  surface: GreenSurface,
): PuttScenario {
  const libraryName = head.provenance.library_name ?? head.name;
  return {
    scenarioId: "react-putting-tab",
    putter: putterSpec(head),
    stroke: puttStroke(clubheadSpeedMps, {
      shaftLeanDeg: setup.shaftLeanDeg,
      aimDeg: setup.aimDeg,
      faceAngleDeg: setup.faceAngleDeg,
      pathAngleDeg: setup.pathAngleDeg,
      attackAngleDeg: setup.attackAngleDeg,
      strikeOffsetToeMm: setup.strikeOffsetToeMm,
      strikeOffsetHighMm: setup.strikeOffsetHighMm,
    }),
    surface,
    stimpFt: setup.stimp,
    holeDistanceM: setup.distance,
    provenance: {
      putterSource: "library",
      putterName: head.name,
      strokeSource: "declared",
      captureModel: setup.captureModel,
      putterMeshSha256: null,
      putterLibraryName: libraryName,
      strokeSourceId: null,
      kernel: PUTTING_RESULT_KERNEL,
    },
    captureModel: setup.captureModel,
    headMoiKgM2: headMoiForStrike(
      head,
      setup.strikeOffsetToeMm,
      setup.strikeOffsetHighMm,
    ),
  };
}

/**
 * The visible playback position, worded exactly as the Qt
 * `PuttPlaybackView` status line so the two surfaces read the same.
 */
function playbackStatus(
  frame: PlaybackFrame | null,
  durationS: number,
  holed: boolean,
): string {
  if (frame === null) return "No putt is loaded for playback.";
  const [xM, yM, zM] = frame.position;
  const outcome = holed && frame.isLanding ? "holed" : "in play";
  return (
    `t ${frame.time.toFixed(3)} s of ${durationS.toFixed(3)} s; ` +
    `x ${xM.toFixed(3)} m; y ${yM.toFixed(3)} m; ` +
    `elevation ${zM.toFixed(3)} m; ${outcome}.`
  );
}

interface PuttingPanelProps {
  onGlossary?: (term: string) => void;
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
  /** Production computation authority; injectable for failure tests. */
  executeStudy?: typeof evaluatePuttWithTrajectory;
}

interface AcceptedStudy {
  executor: typeof evaluatePuttWithTrajectory;
  result: PuttResult;
  /**
   * The very green this result was integrated on. Playback reads its
   * elevations off the scenario's own surface rather than re-deriving an
   * equal one, so the replayed ball can never ride a different green.
   */
  surface: GreenSurface;
  document: PuttingResultDocument;
  twist: PutterTwist;
  plan: PuttingSamplePlan;
  context: string;
  holeX: number;
  grade: number;
  aspect: number;
}

export function PuttingPanel({
  onGlossary,
  distanceUnit = "yd",
  executeStudy = evaluatePuttWithTrajectory,
}: PuttingPanelProps) {
  const formatM = (value: number) => formatDistance(value, distanceUnit);
  const putters = useMemo(putterChoices, []);
  const [setup, setSetup] = useState<PuttSetup>(() => ({
    ...DEFAULT_PUTT_SETUP,
    putterName: putters[0].name,
  }));
  const [explained, setExplained] = useState(PUTTING_RESULT_ROWS[0].key);
  const [selection, setSelection] = useState<{
    accepted: AcceptedStudy;
    rawIndex: number;
  } | null>(null);
  const acceptedStudy = useRef<AcceptedStudy | null>(null);
  const [playbackTimeS, setPlaybackTimeS] = useState(0);
  /**
   * A heightfield imported through the green-import action (ADR-0045
   * F2), replacing the planar grade/aspect green outright — the React
   * analogue of the Qt `PuttingGreenControls._imported` field.
   */
  const [importedGreen, setImportedGreen] = useState<ImportedGreenState | null>(
    null,
  );
  const [importErrorMessage, setImportErrorMessage] = useState("");

  const candidate = useMemo(() => {
    const head =
      putters.find((putter) => putter.name === setup.putterName) ?? putters[0];
    try {
      const clubheadSpeed =
        setup.paceMode === "backstroke"
          ? clubheadSpeedFromBackstroke(setup.backstrokeCm / 100)
          : setup.speed;
      const surface =
        importedGreen?.surface ?? planarSurface(setup.grade, setup.aspect);
      const scenario = puttScenario(head, clubheadSpeed, setup, surface);
      const evaluated = executeStudy(scenario);
      const result = snapshotPuttingResult(evaluated.result);
      const plan = planPuttingSamples(puttingSampleSource(result));
      validatePuttingResultSummary(result, plan);
      const twist = twistResponse(head, clubheadSpeed, {
        shaftLeanDeg: setup.shaftLeanDeg,
        attackAngleDeg: setup.attackAngleDeg,
        strikeOffsetToeMm: setup.strikeOffsetToeMm,
        strikeOffsetHighMm: setup.strikeOffsetHighMm,
      });
      const context = puttingContextLabel(
        putterSpec(head),
        clubheadSpeed,
        setup.stimp,
        setup.grade,
        setup.aspect,
        setup.distance,
      );
      const accepted: AcceptedStudy = {
        executor: executeStudy,
        result,
        surface: scenario.surface,
        document: Object.freeze({ ...evaluated.document }),
        twist,
        plan,
        context,
        holeX: setup.distance,
        grade: setup.grade,
        aspect: setup.aspect,
      };
      return { accepted, error: null };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { accepted: null, error: message.slice(0, 512) };
    }
  }, [executeStudy, putters, setup, importedGreen]);

  const accepted =
    candidate.accepted ??
    (acceptedStudy.current?.executor === executeStudy
      ? acceptedStudy.current
      : null);
  useLayoutEffect(() => {
    if (candidate.accepted !== null) acceptedStudy.current = candidate.accepted;
  }, [candidate]);
  const { error } = candidate;
  const result = accepted?.result ?? null;
  const plan = accepted?.plan ?? null;
  const timeline = useMemo(
    () =>
      accepted === null
        ? null
        : new PlaybackTimeline(
            puttPlaybackSamples(accepted.result, accepted.surface),
          ),
    [accepted],
  );
  useEffect(() => setPlaybackTimeS(0), [timeline]);
  const playbackDurationS = timeline?.duration ?? 0;
  const playbackFrame = timeline?.frameAt(playbackTimeS) ?? null;
  const selectedRawIndex =
    selection?.accepted === accepted ? selection.rawIndex : null;
  const selectSample = (rawIndex: number | null) => {
    setSelection(
      rawIndex === null || accepted === null
        ? null
        : { accepted, rawIndex },
    );
  };

  const values: Record<string, string> = accepted
    ? puttingResultValues(accepted.document, accepted.twist, formatM)
    : {};
  const explainedRow =
    PUTTING_RESULT_ROWS.find((row) => row.key === explained) ??
    PUTTING_RESULT_ROWS[0];
  const explainedTerm =
    explainedRow.term !== undefined && GLOSSARY[explainedRow.term] !== undefined
      ? explainedRow.term
      : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <section aria-label="Putt setup" className="space-y-4">
        <PuttingControls
          setup={setup}
          putters={putters}
          onChange={(patch) => setSetup((current) => ({ ...current, ...patch }))}
          importedGreen={importedGreen}
          importErrorMessage={importErrorMessage}
          onGreenImported={(state) => {
            setImportedGreen(state);
            setImportErrorMessage("");
          }}
          onGreenImportError={setImportErrorMessage}
          onUsePlanarGreen={() => {
            setImportedGreen(null);
            setImportErrorMessage("");
          }}
        />

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Putt Results
          </h2>
          {PUTTING_RESULT_ROWS.map((row) => (
            <button
              key={row.key}
              type="button"
              onClick={() => setExplained(row.key)}
              aria-pressed={explained === row.key}
              title={`Click for a plain-language explanation of ${row.label}`}
              className={
                "mb-1 flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-1.5 text-left text-sm transition-all " +
                (explained === row.key
                  ? "border-sky-400/60 bg-sky-500/10 ring-1 ring-sky-400/40"
                  : "border-slate-800 bg-slate-900/40 hover:border-slate-600")
              }
            >
              <span className="text-slate-300">{row.label}</span>
              <span className="text-right font-semibold text-slate-100">
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
          {explainedTerm !== null ? (
            <button
              type="button"
              title="Open the Glossary at the matching term"
              onClick={() => onGlossary?.(explainedTerm)}
              className="mt-2 text-sky-400 hover:text-sky-300"
            >
              Glossary
            </button>
          ) : null}
        </div>
      </section>

      <section
        aria-label="Green view"
        className="order-first space-y-4 lg:order-none"
      >
        {error ? (
          <p
            role="alert"
            className="rounded border border-red-500/60 bg-red-950/70 px-3 py-2 text-sm text-red-100"
          >
            Attempted putting configuration rejected: {error}.{" "}
            {accepted
              ? "The accepted context below remains displayed."
              : "No accepted putt is available."}
          </p>
        ) : null}
        {accepted ? (
          <p
            aria-label="Displayed putting result context"
            className="text-xs text-slate-400"
          >
            Displayed result: {accepted.context}
          </p>
        ) : null}
        <PlaybackTransportBar
          subjectLabel="Putt"
          subjectPhrase="putt"
          timeS={playbackTimeS}
          durationS={playbackDurationS}
          events={[
            { label: "Strike", timeS: 0 },
            { label: "Finish", timeS: playbackDurationS },
          ]}
          scrubTitle="Physical putt time [s] from strike to rest or capture, interpolated between the retained integrator samples"
          onTimeChange={setPlaybackTimeS}
        />
        <p
          role="status"
          aria-label="Putt playback frame"
          className="rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-200"
        >
          {playbackStatus(
            playbackFrame,
            playbackDurationS,
            result?.holed ?? false,
          )}
        </p>
        <PuttingVisuals
          result={result}
          plan={plan}
          document={accepted?.document ?? null}
          selectedRawIndex={selectedRawIndex}
          onSelectionChange={selectSample}
          holeX={accepted?.holeX ?? setup.distance}
          grade={accepted?.grade ?? setup.grade}
          aspect={accepted?.aspect ?? setup.aspect}
          playbackPositionM={playbackFrame?.position ?? null}
        />
      </section>
    </div>
  );
}
