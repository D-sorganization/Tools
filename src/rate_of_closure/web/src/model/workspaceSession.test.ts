import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DEFAULT_SCENARIO } from "./impact";
import { DEFAULT_PRIMARY_VIEW_STATE } from "./viewPreferences";
import { defaultViewWorkspace } from "./viewWorkspace";
import {
  createWorkspaceDocument,
  parseWorkspaceDocument,
  type WorkspaceSessionSnapshot,
} from "./workspaceSession";

const snapshot = (): WorkspaceSessionSnapshot => ({
  scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -900 },
  club: getClub("Driver 10.5°"),
  units: { speed: "mph", rotation: "deg/s", length: "mm", distance: "yd" },
  modules: DEFAULT_PRIMARY_VIEW_STATE,
  viewWorkspace: defaultViewWorkspace,
});

const metadata = {
  documentId: "workspace.web.test",
  title: "Web test",
  createdAtUtc: "2026-08-10T12:00:00Z",
  modifiedAtUtc: "2026-08-10T12:01:00Z",
  appVersion: "1.14.30",
};

describe("whole workspace session contract", () => {
  it("round trips the supported live explorer state", () => {
    const encoded = createWorkspaceDocument(snapshot(), metadata);
    expect(parseWorkspaceDocument(encoded)).toEqual(snapshot());
    expect(JSON.parse(encoded).schema_version).toBe(2);
  });

  it("rejects unsupported domain state before returning applicable values", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.variation_plan = { schema_version: 2 };
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(/variation/i);
  });

  it("rejects corrupt module and compositor documents", () => {
    const missingModule = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    missingModule.layout.module_order = ["explorer"];
    expect(() => parseWorkspaceDocument(JSON.stringify(missingModule))).toThrow(/module/i);

    const futureView = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    futureView.layout.view_workspace.data.format = "rate_of_closure.view_workspace/9";
    expect(() => parseWorkspaceDocument(JSON.stringify(futureView))).toThrow(/format/i);
  });

  it("matches the native stable identity and strict UTC metadata boundary", () => {
    const localTime = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    localTime.metadata.created_at_utc = "2026-08-10T12:00:00-07:00";
    expect(() => parseWorkspaceDocument(JSON.stringify(localTime))).toThrow(/UTC/i);

    const unstableId = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    unstableId.metadata.document_id = "workspace id with spaces";
    expect(() => parseWorkspaceDocument(JSON.stringify(unstableId))).toThrow(/identifier/i);
  });
});
