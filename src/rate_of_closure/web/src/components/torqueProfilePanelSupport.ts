import {
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  type DoublePendulumRunConfig,
} from "../model/doublePendulum";
import type { SimulationRunTs, WebSourceKind } from "../model/simulation";
import {
  loadTorqueProfileLibrary,
  starterTorqueProfile,
} from "../model/torqueProfileEditor";
import type { PrescribedTorqueProfile } from "../model/torqueProfiles";

export interface TorqueProfileEditorState {
  profileId: string;
  name: string;
  description: string;
  startS: string;
  endS: string;
  shoulder: string;
  wrist: string;
}

export interface TorqueProfilePanelProps {
  sourceKind: WebSourceKind;
  runConfig: DoublePendulumRunConfig;
  onRunConfigChange: (config: DoublePendulumRunConfig) => void;
  storage?: Storage;
  run?: SimulationRunTs | null;
  profiles?: readonly PrescribedTorqueProfile[];
  activeProfileId?: string | null;
  onLibraryChange?: (
    profiles: readonly PrescribedTorqueProfile[],
    activeProfileId: string,
  ) => void;
}

export function representativeSubset<T>(
  values: readonly T[],
  limit: number,
): readonly T[] {
  if (values.length <= limit) return values;
  return Object.freeze(Array.from({ length: limit }, (_, index) =>
    values[Math.round((index * (values.length - 1)) / (limit - 1))]));
}

export function displayNumber(value: number): string {
  return String(Number(value.toFixed(3)));
}

function coefficients(
  profile: PrescribedTorqueProfile,
  jointId: string,
): string {
  const values = profile.assignments.find((item) => item.jointId === jointId)
    ?.polynomial.coefficients ?? [];
  return values.join(", ");
}

export function editorFor(
  profile: PrescribedTorqueProfile,
): TorqueProfileEditorState {
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

export function initialProfiles(
  storage?: Storage,
): readonly PrescribedTorqueProfile[] {
  try {
    return storage ? loadTorqueProfileLibrary(storage) : loadTorqueProfileLibrary();
  } catch {
    return Object.freeze([starterTorqueProfile()]);
  }
}

export function formatFit(values: readonly number[]): string {
  return values.map((value) => String(Number(value.toPrecision(10)))).join(", ");
}

export async function fileText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}
