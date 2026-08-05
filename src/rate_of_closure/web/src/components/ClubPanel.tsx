/**
 * Club group — library picker, loft override, bulge & roll toggle, and
 * the "Generate Representative Head" action. Mirrors the PyQt6 Club
 * group: selecting a club drives GC-to-face and lie from its spec
 * (overrides stay editable), and generation builds the parametric head
 * client-side into the existing mesh render path.
 */

import { useEffect, useState } from "react";

import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";
import {
  CLUB_LIBRARY,
  getClub,
  type ClubSpec,
} from "../model/club";
import {
  generatedHeadFor,
  type GeneratedHead,
} from "../model/clubHeadGeneration";
import { FIELD_GUIDANCE } from "../model/units";

const INPUT_CLASS =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 " +
  "py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none " +
  "disabled:opacity-40";

export function ClubPanel({
  onDriveScenario,
  onGenerate,
  onSpecChange,
}: {
  /** Scenario plumbing: adopt the selected club's GC-to-face and lie. */
  onDriveScenario: (comToFaceMm: number, lieAngleDeg: number) => void;
  /** Deliver a generated head with its hosel and volumetric COG. */
  onGenerate: (head: GeneratedHead) => void;
  /** Track the effective club spec (overrides applied) as it changes. */
  onSpecChange?: (spec: ClubSpec) => void;
}) {
  const [clubName, setClubName] = useState<string>(CLUB_LIBRARY[1].name);
  const [loftDeg, setLoftDeg] = useState<number>(CLUB_LIBRARY[1].loftDeg);
  const [curvedFace, setCurvedFace] = useState<boolean>(true);
  const [bulgeMm, setBulgeMm] = useState<number>(300);
  const [rollMm, setRollMm] = useState<number>(280);

  const effectiveSpec = (): ClubSpec => ({
    ...getClub(clubName),
    loftDeg,
    faceBulgeRadiusM: curvedFace ? bulgeMm / 1000 : null,
    faceRollRadiusM: curvedFace ? rollMm / 1000 : null,
  });

  useEffect(() => {
    onSpecChange?.(effectiveSpec());
    // eslint-disable-next-line react-hooks/exhaustive-deps -- state-derived
  }, [clubName, loftDeg, curvedFace, bulgeMm, rollMm]);

  const onClubChange = (name: string) => {
    const club = getClub(name);
    setClubName(name);
    setLoftDeg(club.loftDeg);
    setCurvedFace(club.faceBulgeRadiusM !== null);
    if (club.faceBulgeRadiusM !== null) setBulgeMm(club.faceBulgeRadiusM * 1000);
    if (club.faceRollRadiusM !== null) setRollMm(club.faceRollRadiusM * 1000);
    onDriveScenario(club.cgDepthM * 1000, club.lieDeg);
  };

  const onGenerateHead = () => {
    const spec = effectiveSpec();
    onGenerate(generatedHeadFor(spec));
  };

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Club
      </h2>
      <label title={FIELD_GUIDANCE.clubSelection} className="mb-3 block text-sm">
        <span className="mb-1 block text-slate-300">Club</span>
        <select
          value={clubName}
          onChange={(e) => onClubChange(e.target.value)}
          title={FIELD_GUIDANCE.clubSelection}
          className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
        >
          {CLUB_LIBRARY.map((club) => (
            <option key={club.name} value={club.name}>
              {club.name}
            </option>
          ))}
        </select>
      </label>
      <label title={FIELD_GUIDANCE.clubLoftDeg} className="mb-3 block text-sm">
        <span className="mb-1 flex justify-between text-slate-300">
          <span className="flex items-center">Loft<FieldInfo label="Loft" guidance={FIELD_GUIDANCE.clubLoftDeg} /></span>
          <span className="text-slate-500">deg</span>
        </span>
        <DecimalInput
          step={0.5}
          min={0}
          max={70}
          value={loftDeg}
          aria-label="Loft deg"
          onCommit={setLoftDeg}
          title={FIELD_GUIDANCE.clubLoftDeg}
          className={INPUT_CLASS}
        />
      </label>
      <label
        title={FIELD_GUIDANCE.faceCurvatureEnabled}
        className="mb-3 flex items-center gap-2 text-sm text-slate-300"
      >
        <input
          type="checkbox"
          checked={curvedFace}
          onChange={(e) => setCurvedFace(e.target.checked)}
          title={FIELD_GUIDANCE.faceCurvatureEnabled}
        />
        Curved Face (Bulge &amp; Roll)
      </label>
      {(
        [
          ["Bulge Radius", bulgeMm, setBulgeMm, "faceBulgeRadiusMm"],
          ["Roll Radius", rollMm, setRollMm, "faceRollRadiusMm"],
        ] as const
      ).map(([label, value, setValue, guidanceKey]) => (
        <label
          key={label}
          title={FIELD_GUIDANCE[guidanceKey]}
          className="mb-3 block text-sm"
        >
          <span className="mb-1 flex justify-between text-slate-300">
            <span className="flex items-center">{label}<FieldInfo label={label} guidance={FIELD_GUIDANCE[guidanceKey]} /></span>
            <span className="text-slate-500">mm</span>
          </span>
          <DecimalInput
            step={10}
            min={100}
            max={2000}
            value={value}
            disabled={!curvedFace}
            aria-label={`${label} mm`}
            onCommit={setValue}
            title={FIELD_GUIDANCE[guidanceKey]}
            className={INPUT_CLASS}
          />
        </label>
      ))}
      <button
        type="button"
        onClick={onGenerateHead}
        title="Build a parametric head mesh from the selected club spec (loft, mass envelope, bulge & roll) and render it in the 3D view in place of the wireframe."
        className="w-full rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1.5 text-sm font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        Generate Representative Head
      </button>
    </div>
  );
}
