import { useMemo, useState } from "react";

import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  prescribedDoublePendulumRun,
  DOUBLE_PENDULUM_MODEL_ID,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  type DoublePendulumRunConfig,
} from "../model/doublePendulum";
import {
  fitTorqueRows,
  loadTorqueProfileLibrary,
  parseCoefficientText,
  parseTorqueSampleRows,
  saveTorqueProfileLibrary,
  starterTorqueProfile,
  type TorqueFit,
} from "../model/torqueProfileEditor";
import {
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TorquePolynomial,
  TorqueProfileSource,
} from "../model/torqueProfiles";
import { type WebSourceKind } from "../model/simulation";
import { type SimulationRunTs } from "../model/simulation";
import { TorqueFitPreview } from "./TorqueFitPreview";

interface Props {
  sourceKind: WebSourceKind;
  runConfig: DoublePendulumRunConfig;
  onRunConfigChange: (config: DoublePendulumRunConfig) => void;
  storage?: Storage;
  run?: SimulationRunTs | null;
}

interface EditorState {
  profileId: string;
  name: string;
  description: string;
  startS: string;
  endS: string;
  shoulder: string;
  wrist: string;
}

const INPUT =
  "w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none";
const SAMPLE_ROWS = "0, 18, -4\n0.75, 9, 0.5\n1.5, 0, 5";
const MAX_EDITABLE_SAMPLE_ROWS = 101;
const MAX_TABLE_ROWS = 25;

function representativeSubset<T>(values: readonly T[], limit: number): readonly T[] {
  if (values.length <= limit) return values;
  return Object.freeze(Array.from({ length: limit }, (_, index) =>
    values[Math.round((index * (values.length - 1)) / (limit - 1))]));
}

function coefficients(profile: PrescribedTorqueProfile, jointId: string): string {
  const values = profile.assignments.find((item) => item.jointId === jointId)
    ?.polynomial.coefficients ?? [];
  return values.join(", ");
}

function editorFor(profile: PrescribedTorqueProfile): EditorState {
  return {
    profileId: profile.profileId,
    name: profile.name,
    description: profile.description,
    startS: String(profile.timeDomainS[0]),
    endS: String(profile.timeDomainS[1]),
    shoulder: coefficients(profile, SHOULDER_JOINT_ID),
    wrist: coefficients(profile, WRIST_JOINT_ID),
  };
}

function initialProfiles(storage?: Storage): readonly PrescribedTorqueProfile[] {
  try {
    return storage ? loadTorqueProfileLibrary(storage) : loadTorqueProfileLibrary();
  } catch {
    return Object.freeze([starterTorqueProfile()]);
  }
}

function formatFit(values: readonly number[]): string {
  return values.map((value) => String(Number(value.toPrecision(10)))).join(", ");
}

async function fileText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

export function TorqueProfilePanel({
  sourceKind,
  runConfig,
  onRunConfigChange,
  storage,
  run = null,
}: Props) {
  const [profiles, setProfiles] = useState(() => initialProfiles(storage));
  const [selectedId, setSelectedId] = useState(profiles[0].profileId);
  const [editor, setEditor] = useState(() => editorFor(profiles[0]));
  const [sampleText, setSampleText] = useState(SAMPLE_ROWS);
  const [fitDegree, setFitDegree] = useState(2);
  const [fit, setFit] = useState<TorqueFit | null>(null);
  const [fitOrigin, setFitOrigin] = useState<"drawn" | "fitted_run">("drawn");
  const [sourceSampleCount, setSourceSampleCount] = useState<number | null>(null);
  const [message, setMessage] = useState("Ready — changes are not saved yet.");
  const selected = profiles.find((profile) => profile.profileId === selectedId)
    ?? profiles[0];
  const prescribedAvailable = sourceKind === "double_pendulum";

  const previewRows = useMemo(() => {
    try {
      return parseTorqueSampleRows(sampleText);
    } catch {
      return [];
    }
  }, [sampleText]);
  const visiblePreviewRows = useMemo(
    () => representativeSubset(previewRows, MAX_TABLE_ROWS),
    [previewRows],
  );

  const selectProfile = (profileId: string) => {
    const profile = profiles.find((item) => item.profileId === profileId);
    if (!profile) return;
    setSelectedId(profileId);
    setEditor(editorFor(profile));
    setFit(null);
    setMessage(`Loaded ${profile.name} into the editor.`);
    if (runConfig.mode === "prescribed") {
      onRunConfigChange(prescribedDoublePendulumRun(profile));
    }
  };

  const selectMode = (mode: string) => {
    if (mode === "passive") {
      onRunConfigChange(PASSIVE_DOUBLE_PENDULUM_RUN);
    } else if (prescribedAvailable) {
      onRunConfigChange(prescribedDoublePendulumRun(selected));
    }
  };

  const edit = (key: keyof EditorState, value: string) => {
    setEditor((current) => ({ ...current, [key]: value }));
    if (key === "shoulder" || key === "wrist") setFit(null);
  };

  const saveProfile = () => {
    try {
      const shoulder = fit?.shoulder
        ?? new TorquePolynomial(parseCoefficientText(editor.shoulder));
      const wrist = fit?.wrist
        ?? new TorquePolynomial(parseCoefficientText(editor.wrist));
      const now = new Date().toISOString();
      const previous = profiles.find((item) => item.profileId === editor.profileId);
      const profile = new PrescribedTorqueProfile({
        profileId: editor.profileId,
        modelId: DOUBLE_PENDULUM_MODEL_ID,
        name: editor.name,
        description: editor.description,
        source: fit
          ? fitOrigin === "fitted_run" ? TorqueProfileSource.FITTED_RUN : TorqueProfileSource.DRAWN
          : TorqueProfileSource.DIRECT,
        sourceMetadata: {
          author: "web-editor",
          workflow: fit
            ? `${fitOrigin}_polynomial_degree_${fitDegree}`
            : "direct_coefficients",
          ...(fitOrigin === "fitted_run" && run ? {
            run_profile_id: run.torqueRun.profileId ?? "passive",
            impact_outcome: run.impactOutcome.status,
          } : {}),
        },
        createdAtUtc: previous?.createdAtUtc ?? now,
        modifiedAtUtc: now,
        timeDomainS: [Number(editor.startS), Number(editor.endS)],
        assignments: [
          new JointTorqueAssignment(SHOULDER_JOINT_ID, shoulder),
          new JointTorqueAssignment(WRIST_JOINT_ID, wrist),
        ],
      });
      const updated = [...profiles.filter((item) => item.profileId !== profile.profileId), profile];
      if (storage) saveTorqueProfileLibrary(updated, storage);
      else saveTorqueProfileLibrary(updated);
      setProfiles(Object.freeze(updated));
      setSelectedId(profile.profileId);
      setEditor(editorFor(profile));
      if (runConfig.mode === "prescribed") {
        onRunConfigChange(prescribedDoublePendulumRun(profile));
      }
      setMessage(`Saved ${profile.name}; the simulation is now dirty until rerun.`);
    } catch (error) {
      setMessage(`Save failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const fitSamples = () => {
    try {
      const result = fitTorqueRows(sampleText, fitDegree);
      setFit(result);
      setFitOrigin("drawn");
      setSourceSampleCount(null);
      setEditor((current) => ({
        ...current,
        startS: String(result.rows[0].timeS),
        endS: String(result.rows[result.rows.length - 1].timeS),
        shoulder: formatFit(result.shoulder.coefficients),
        wrist: formatFit(result.wrist.coefficients),
      }));
      setMessage(`Degree ${fitDegree} c0-first curves fitted. Save to add them to the library.`);
    } catch (error) {
      setMessage(`Fit failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const fitCurrentRun = () => {
    const history = run?.torqueRun.appliedTorqueHistory ?? [];
    if (history.length < 2) {
      setMessage("Fit failed: the current run has no double-pendulum torque history.");
      return;
    }
    const rows = history.map((sample) => [
      sample.timeS,
      sample.torquesNm[SHOULDER_JOINT_ID],
      sample.torquesNm[WRIST_JOINT_ID],
    ].join(","));
    const text = rows.join("\n");
    setSampleText(representativeSubset(rows, MAX_EDITABLE_SAMPLE_ROWS).join("\n"));
    setSourceSampleCount(history.length);
    try {
      const result = fitTorqueRows(text, fitDegree);
      setFit(result);
      setFitOrigin("fitted_run");
      setEditor((current) => ({
        ...current,
        startS: String(result.rows[0].timeS),
        endS: String(result.rows[result.rows.length - 1].timeS),
        shoulder: formatFit(result.shoulder.coefficients),
        wrist: formatFit(result.wrist.coefficients),
      }));
      setMessage(`Current run fitted at degree ${fitDegree}. Save to create a reusable profile.`);
    } catch (error) {
      setMessage(`Fit failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const importProfile = async (file?: File) => {
    if (!file) return;
    try {
      const profile = PrescribedTorqueProfile.loads(await fileText(file));
      const updated = [...profiles.filter((item) => item.profileId !== profile.profileId), profile];
      if (storage) saveTorqueProfileLibrary(updated, storage);
      else saveTorqueProfileLibrary(updated);
      setProfiles(Object.freeze(updated));
      setSelectedId(profile.profileId);
      setEditor(editorFor(profile));
      setFit(null);
      setMessage(`Imported ${profile.name}.`);
    } catch (error) {
      setMessage(`Import failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const exportProfile = () => {
    const blob = new Blob([selected.dumps()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${selected.profileId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setMessage(`Exported ${selected.name}.`);
  };

  return (
    <section
      aria-label="Prescribed torque profiles"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20"
    >
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
        Prescribed Torque
      </h2>
      <p className="mt-1 text-xs leading-relaxed text-slate-500">
        Explicit shoulder and wrist torques in N·m. Equations use physical time
        and ascending c0-first coefficients: τ(t) = c0 + c1t + c2t² + …
      </p>
      <label className="mt-3 block text-sm text-slate-300">
        Execution Mode
        <select
          aria-label="Torque execution mode"
          value={prescribedAvailable ? runConfig.mode : "passive"}
          onChange={(event) => selectMode(event.target.value)}
          className={`${INPUT} mt-1`}
        >
          <option value="passive">Passive (Default)</option>
          <option value="prescribed" disabled={!prescribedAvailable}>
            Prescribed Profile
          </option>
        </select>
      </label>
      {!prescribedAvailable && (
        <p className="mt-2 text-xs text-amber-300">
          Select Double Pendulum to enable prescribed torque execution.
        </p>
      )}
      <p
        role="status"
        aria-label="Torque execution status"
        className="mt-2 rounded border border-sky-500/25 bg-slate-950/60 p-2 text-xs text-sky-200"
      >
        {runConfig.mode === "prescribed" && prescribedAvailable
          ? `Prescribed — ${selected.name} will drive every RK4 substep.`
          : "Passive — no applied shoulder or wrist torque."}
      </p>

      <label className="mt-3 block text-sm text-slate-300">
        Profile Library
        <select
          aria-label="Torque profile library"
          value={selectedId}
          onChange={(event) => selectProfile(event.target.value)}
          className={`${INPUT} mt-1`}
        >
          {profiles.map((profile) => (
            <option key={profile.profileId} value={profile.profileId}>{profile.name}</option>
          ))}
        </select>
      </label>
      <div className="mt-2 flex flex-wrap gap-2">
        <label className="cursor-pointer rounded border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:border-sky-500">
          Import JSON
          <input
            type="file"
            accept="application/json,.json"
            aria-label="Import torque profile JSON"
            className="sr-only"
            onChange={(event) => void importProfile(event.target.files?.[0])}
          />
        </label>
        <button type="button" title="Download the selected canonical torque profile as JSON" onClick={exportProfile} className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:border-sky-500">
          Export Selected JSON
        </button>
      </div>

      <details className="mt-3" open>
        <summary className="cursor-pointer text-sm font-semibold text-slate-300">
          Equation Editor
        </summary>
        <div className="mt-2 grid gap-2">
          <label className="text-xs text-slate-400">Profile ID<input aria-label="Profile ID" value={editor.profileId} onChange={(event) => edit("profileId", event.target.value)} className={`${INPUT} mt-1`} /></label>
          <label className="text-xs text-slate-400">Name<input aria-label="Profile name" value={editor.name} onChange={(event) => edit("name", event.target.value)} className={`${INPUT} mt-1`} /></label>
          <label className="text-xs text-slate-400">Description<input aria-label="Profile description" value={editor.description} onChange={(event) => edit("description", event.target.value)} className={`${INPUT} mt-1`} /></label>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-xs text-slate-400">Start (s)<input aria-label="Profile start time" type="number" step="0.01" value={editor.startS} onChange={(event) => edit("startS", event.target.value)} className={`${INPUT} mt-1`} /></label>
            <label className="text-xs text-slate-400">End (s)<input aria-label="Profile end time" type="number" step="0.01" value={editor.endS} onChange={(event) => edit("endS", event.target.value)} className={`${INPUT} mt-1`} /></label>
          </div>
          <label className="text-xs text-slate-400">Shoulder c0, c1, …<input aria-label="Shoulder coefficients" value={editor.shoulder} onChange={(event) => edit("shoulder", event.target.value)} className={`${INPUT} mt-1 font-mono`} /></label>
          <label className="text-xs text-slate-400">Wrist c0, c1, …<input aria-label="Wrist coefficients" value={editor.wrist} onChange={(event) => edit("wrist", event.target.value)} className={`${INPUT} mt-1 font-mono`} /></label>
          <button type="button" title="Validate and save this profile in the browser library" onClick={saveProfile} className="rounded border border-emerald-400/60 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-300 hover:bg-emerald-500/20">
            Save Torque Profile
          </button>
        </div>
      </details>

      <details className="mt-3">
        <summary className="cursor-pointer text-sm font-semibold text-slate-300">
          Point Table & Polynomial Fit
        </summary>
        <label className="mt-2 block text-xs text-slate-400">
          Rows: time_s, shoulder_Nm, wrist_Nm
          <textarea aria-label="Torque sample rows" rows={4} value={sampleText} onChange={(event) => {
            setSampleText(event.target.value);
            setFit(null);
            setSourceSampleCount(null);
          }} className={`${INPUT} mt-1 font-mono`} />
        </label>
        <label className="mt-2 block text-xs text-slate-400">
          Polynomial Degree
          <select aria-label="Polynomial degree" value={fitDegree} onChange={(event) => setFitDegree(Number(event.target.value))} className={`${INPUT} mt-1`}>
            {[0, 1, 2, 3].map((degree) => <option key={degree} value={degree}>{degree}</option>)}
          </select>
        </label>
        {previewRows.length > 0 && (
          <>
          <p role="status" aria-label="Torque sample display status" className="mt-2 text-xs text-slate-500">
            {sourceSampleCount !== null && sourceSampleCount > previewRows.length
              ? `Editor contains ${previewRows.length} representative rows from ${sourceSampleCount} fitted samples; `
              : ""}
            Table showing {visiblePreviewRows.length} of {previewRows.length} editor rows.
          </p>
          <table aria-label="Torque sample preview" className="mt-2 w-full text-right text-xs text-slate-300">
            <thead><tr><th>Time (s)</th><th>Shoulder</th><th>Wrist</th></tr></thead>
            <tbody>{visiblePreviewRows.map((row) => (
              <tr key={row.timeS}><td>{row.timeS}</td><td>{row.shoulderNm}</td><td>{row.wristNm}</td></tr>
            ))}</tbody>
          </table>
          </>
        )}
        <button type="button" title="Fit the editable torque points using the selected polynomial degree" onClick={fitSamples} className="mt-2 w-full rounded border border-violet-400/60 bg-violet-500/10 px-3 py-2 text-sm font-semibold text-violet-300 hover:bg-violet-500/20">
          Fit Torque Curves
        </button>
        <button type="button" title="Fit the current run's retained torque history into a reusable profile" onClick={fitCurrentRun} disabled={!run || run.torqueRun.appliedTorqueHistory.length < 2} className="mt-2 w-full rounded border border-slate-600 px-3 py-2 text-sm text-slate-300 disabled:opacity-40">
          Fit Current Run to Profile
        </button>
        <TorqueFitPreview fit={fit} />
      </details>
      <p role="status" aria-label="Torque profile editor status" className="mt-3 text-xs text-slate-400">
        {message}
      </p>
    </section>
  );
}
