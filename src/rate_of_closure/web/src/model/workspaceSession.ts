/** Strict whole-workspace adapter for the browser's supported live state. */

import type { UnitSelections } from "../components/ImpactExplorerPanel";
import type { ClubSpec } from "./club";
import type { ImpactScenario } from "./impact";
import type { PrimaryViewState } from "./viewPreferences";
import {
  viewWorkspaceDocument,
  viewWorkspaceFromDocument,
  type ViewWorkspace,
} from "./viewWorkspace";
import {
  simulationWorkspaceDocument,
  simulationWorkspaceFromSession,
  type SimulationWorkspaceSnapshot,
} from "./workspaceSimulationSession";
import {
  torqueWorkspaceDocument,
  torqueWorkspaceFromSession,
  type TorqueWorkspaceSnapshot,
} from "./workspaceTorqueSession";
import {
  migratedLegacyVariationFallback,
  variationPlanFromWorkspaceDocument,
  variationPlanWorkspaceDocument,
  variationWorkspaceDocument,
  variationWorkspaceFromDocument,
  type VariationWorkspaceSnapshot,
} from "./workspaceVariationSession";
import {
  validateWorkspaceMetadata,
  versionedPayload,
} from "./workspaceMetadataValidation";
import {
  capabilityWorkflowDocument,
  capabilityWorkflowFromDocument,
  capabilityWorkflowInputs,
  type CapabilityWorkflowDocument,
} from "./capabilityWorkflow";
import {
  validatedWorkspaceModules,
  validatedWorkspaceUnits,
  workspaceClubDocument,
  workspaceClubFromDocument,
  workspaceScenarioDocument,
  workspaceScenarioFromDocument,
} from "./workspaceSessionFields";

const WORKSPACE_SCHEMA = "rate_of_closure.workspace";
const WORKSPACE_VERSION = 3;
const SESSION_SCHEMA = "rate_of_closure.explorer_session";
const CLUB_SCHEMA = "rate_of_closure.club_configuration";
const SESSION_PAYLOAD_VERSION = 5;
const CLUB_PAYLOAD_VERSION = 1;

export interface WorkspaceSessionSnapshot {
  readonly scenario: ImpactScenario;
  readonly club: ClubSpec;
  readonly units: UnitSelections;
  readonly simulation: SimulationWorkspaceSnapshot;
  readonly torque: TorqueWorkspaceSnapshot;
  readonly variation: VariationWorkspaceSnapshot;
  readonly capability: CapabilityWorkflowDocument;
  readonly modules: PrimaryViewState;
  readonly viewWorkspace: ViewWorkspace;
}

export interface WorkspaceFileMetadata {
  readonly documentId: string;
  readonly title: string;
  readonly createdAtUtc: string;
  readonly modifiedAtUtc: string;
  readonly appVersion: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function exactRecord(
  value: unknown,
  keys: readonly string[],
  context: string,
): Record<string, unknown> {
  if (!isRecord(value)) throw new TypeError(`${context} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== keys.length || keys.some((key) => !(key in value))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return value;
}

/** Serialize the browser-supported whole workspace without hidden file authority. */
export function createWorkspaceDocument(
  snapshot: WorkspaceSessionSnapshot,
  metadata: WorkspaceFileMetadata,
): string {
  const document = {
    schema: WORKSPACE_SCHEMA,
    schema_version: WORKSPACE_VERSION,
    metadata: {
      document_id: metadata.documentId,
      title: metadata.title,
      created_at_utc: metadata.createdAtUtc,
      modified_at_utc: metadata.modifiedAtUtc,
      app_version: metadata.appVersion,
      provenance: { surface: "rate-of-closure-file-adapter/v1" },
    },
    model_session: {
      schema: SESSION_SCHEMA,
      schema_version: SESSION_PAYLOAD_VERSION,
      data: {
        scenario: workspaceScenarioDocument(snapshot.scenario),
        units: snapshot.units,
        simulation_setup: simulationWorkspaceDocument(
          snapshot.simulation,
          snapshot.club,
        ),
        torque_selection: torqueWorkspaceDocument(snapshot.torque),
        variation_study: variationWorkspaceDocument(
          snapshot.variation,
          snapshot.simulation.ballSetup,
        ),
        capability_request: capabilityWorkflowDocument(snapshot.capability),
      },
    },
    prescribed_torque_profiles: snapshot.torque.profiles.map((profile) =>
      profile.toJsonObject(),
    ),
    club_configuration: {
      schema: CLUB_SCHEMA,
      schema_version: CLUB_PAYLOAD_VERSION,
      data: workspaceClubDocument(snapshot.club),
    },
    variation_plan: variationPlanWorkspaceDocument(
      snapshot.variation,
      snapshot.simulation.ballSetup,
    ),
    layout: {
      module_order: snapshot.modules.order,
      visible_module_ids: snapshot.modules.visible,
      active_module_id: snapshot.modules.active,
      view_workspace: {
        schema: "rate_of_closure.view_workspace",
        schema_version: 2,
        data: viewWorkspaceDocument(snapshot.viewWorkspace),
      },
    },
  };
  parseWorkspaceDocument(JSON.stringify(document));
  return `${JSON.stringify(document, null, 2)}\n`;
}

/** Parse and validate the entire document before exposing any applicable state. */
export interface WorkspaceParseOptions {
  readonly legacySimulationFallback?: SimulationWorkspaceSnapshot;
  readonly legacyTorqueFallback?: TorqueWorkspaceSnapshot;
  readonly legacyVariationFallback?: VariationWorkspaceSnapshot;
  readonly legacyCapabilityFallback?: CapabilityWorkflowDocument;
}

/** Parse a current file or deliberately migrate v1 with an explicit fallback. */
export function parseWorkspaceDocument(
  text: string,
  options: WorkspaceParseOptions = {},
): WorkspaceSessionSnapshot {
  if (typeof text !== "string" || text.trim().length === 0) {
    throw new TypeError("workspace file must contain JSON text");
  }
  const root = exactRecord(
    JSON.parse(text),
    [
      "schema",
      "schema_version",
      "metadata",
      "model_session",
      "prescribed_torque_profiles",
      "club_configuration",
      "variation_plan",
      "layout",
    ],
    "workspace",
  );
  if (
    root.schema !== WORKSPACE_SCHEMA ||
    (root.schema_version !== 2 && root.schema_version !== WORKSPACE_VERSION)
  ) {
    throw new TypeError("unsupported workspace schema");
  }
  validateWorkspaceMetadata(root.metadata);
  const sessionEnvelope = versionedPayload(
    root.model_session,
    SESSION_SCHEMA,
    [1, 2, 3, 4, SESSION_PAYLOAD_VERSION],
    "model_session",
  );
  const session = exactRecord(
    sessionEnvelope.data,
    sessionEnvelope.version === 1
      ? ["scenario", "units"]
      : sessionEnvelope.version === 2
        ? ["scenario", "units", "simulation_setup"]
        : sessionEnvelope.version === 3
          ? ["scenario", "units", "simulation_setup", "torque_selection"]
        : sessionEnvelope.version === 4
          ? [
              "scenario",
              "units",
              "simulation_setup",
              "torque_selection",
              "variation_study",
            ]
          : [
              "scenario",
              "units",
              "simulation_setup",
              "torque_selection",
              "variation_study",
              "capability_request",
            ],
    "model_session.data",
  );
  const clubEnvelope = versionedPayload(
    root.club_configuration,
    CLUB_SCHEMA,
    [CLUB_PAYLOAD_VERSION],
    "club_configuration",
  );
  const club = exactRecord(
    clubEnvelope.data,
    Object.keys(
      workspaceClubDocument({
        name: "x",
        clubType: "Driver",
        lengthM: 1,
        headMassKg: 0.2,
        loftDeg: 10,
        lieDeg: 56,
        moiAboutShaftKgM2: 0.0005,
        cgDepthM: 0.02,
        cgHeightM: 0.02,
        faceBulgeRadiusM: null,
        faceRollRadiusM: null,
      }),
    ),
    "club_configuration.data",
  );
  const layout = exactRecord(
    root.layout,
    [
      "module_order",
      "visible_module_ids",
      "active_module_id",
      "view_workspace",
    ],
    "layout",
  );
  const viewEnvelope = exactRecord(
    layout.view_workspace,
    ["schema", "schema_version", "data"],
    "layout.view_workspace",
  );
  if (
    viewEnvelope.schema !== "rate_of_closure.view_workspace" ||
    (viewEnvelope.schema_version !== 1 && viewEnvelope.schema_version !== 2)
  ) {
    throw new TypeError("unsupported view workspace payload");
  }
  const viewData = exactRecord(
    viewEnvelope.data,
    viewEnvelope.schema_version === 1
      ? ["format", "layout", "slots", "active_slot_id", "playback"]
      : [
          "format",
          "layout",
          "slots",
          "active_slot_id",
          "playback",
          "camera_preferences",
        ],
    "layout.view_workspace.data",
  );
  if (
    (viewEnvelope.schema_version === 1 &&
      viewData.format !== "rate_of_closure.view_workspace/1") ||
    (viewEnvelope.schema_version === 2 &&
      viewData.format !== "rate_of_closure.view_workspace/2")
  ) {
    throw new TypeError("view workspace envelope version does not match its format");
  }
  const parsedClub = workspaceClubFromDocument(club);
  const simulation: SimulationWorkspaceSnapshot =
    simulationWorkspaceFromSession({
      isLegacy: sessionEnvelope.version === 1,
      setupDocument: session.simulation_setup,
      club: parsedClub,
      legacyFallback: options.legacySimulationFallback,
    });
  const torque: TorqueWorkspaceSnapshot = torqueWorkspaceFromSession({
    isLegacy: sessionEnvelope.version < 3,
    selectionDocument: session.torque_selection,
    profileDocuments: root.prescribed_torque_profiles,
    legacyFallback: options.legacyTorqueFallback,
  });
  const documentPlan =
    root.variation_plan === null
      ? null
      : variationPlanFromWorkspaceDocument(
          root.variation_plan,
          simulation.ballSetup,
          root.schema_version === 2,
        );
  let variation: VariationWorkspaceSnapshot;
  if (sessionEnvelope.version < 4) {
    if (options.legacyVariationFallback === undefined) {
      throw new RangeError(
        "legacy model_session requires an explicit variation migration fallback",
      );
    }
    variation = migratedLegacyVariationFallback(
      options.legacyVariationFallback,
          documentPlan,
      simulation.ballSetup,
    );
  } else {
    if (documentPlan === null) {
      throw new RangeError(
        "current workspace requires a canonical variation plan",
      );
    }
    variation = variationWorkspaceFromDocument(
      session.variation_study,
      documentPlan.plan,
      documentPlan,
      simulation.ballSetup,
    );
  }
  let capability: CapabilityWorkflowDocument;
  if (sessionEnvelope.version < SESSION_PAYLOAD_VERSION) {
    if (options.legacyCapabilityFallback === undefined) {
      throw new RangeError(
        "legacy model_session requires an explicit capability migration fallback",
      );
    }
    capabilityWorkflowInputs(options.legacyCapabilityFallback);
    capability = options.legacyCapabilityFallback;
  } else {
    capability = capabilityWorkflowFromDocument(session.capability_request);
    capabilityWorkflowInputs(capability);
  }
  return {
    scenario: workspaceScenarioFromDocument(session.scenario),
    club: parsedClub,
    units: validatedWorkspaceUnits(session.units),
    simulation,
    torque,
    variation,
    capability,
    modules: validatedWorkspaceModules(layout),
    viewWorkspace: viewWorkspaceFromDocument(viewData),
  };
}
