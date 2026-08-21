import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { LaunchMonitorPlayerWorkspace } from "./LaunchMonitorPlayerWorkspace";

const rows: LaunchMonitorRow[] = Array.from({ length: 12 }, (_, index) => ({
  shot_id: `s${index}`,
  player_id: index < 6 ? "p1" : "p2",
  face_angle: index,
  club_path: index * 0.8 + (index % 2),
}));

describe("LaunchMonitorPlayerWorkspace", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("fails closed until a player identity column is explicitly attested", async () => {
    render(<LaunchMonitorPlayerWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.getByRole("button", { name: /run offline compatibility covariation/i })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Player identity column"), { target: { value: "player_id" } });
    fireEvent.click(screen.getByLabelText(/I attest/i));
    await waitFor(() => expect(
      screen.getByRole("button", { name: /run offline compatibility covariation/i }),
    ).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /run offline compatibility covariation/i }));
    await waitFor(() => expect(screen.getByText(/2 player groups analyzed/i)).toBeInTheDocument());
  });

  it("offers reference-only project persistence and explicit full export", () => {
    render(<LaunchMonitorPlayerWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.getByRole("button", { name: /save project/i })).toHaveAttribute("title", expect.stringMatching(/does not embed/i));
    expect(screen.getByRole("button", { name: /export full bundle/i })).toHaveAttribute("title", expect.stringMatching(/backing rows/i));
    expect(screen.getByLabelText("Load saved launch-monitor project")).toBeInTheDocument();
  });

  it("locks advanced population analysis to the attested identity", () => {
    render(<LaunchMonitorPlayerWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.queryByLabelText("Covariation player column")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Player identity column"), {
      target: { value: "player_id" },
    });
    fireEvent.click(screen.getByLabelText(/I attest/i));
    expect(screen.getByLabelText("Covariation player column")).toHaveValue("player_id");
    expect(screen.getByLabelText("Covariation player column")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Analyze Player Covariation" })).toHaveAttribute("title");
  });

  it("offers fail-closed canonical authority and authorized corpus controls", () => {
    render(<LaunchMonitorPlayerWorkspace rows={rows} sourceName="test.csv" />);
    expect(screen.getByLabelText("Canonical Upstream authority URL")).toBeInTheDocument();
    expect(screen.getByLabelText("Load authorized corpus reference")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /inspect authorized corpus/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /run canonical player covariation/i })).toBeDisabled();
    expect(screen.getByText(/canonical inline limit is 20,000 rows/i)).toBeInTheDocument();
  });

  it("makes validated canonical evidence inspectable and exportable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        contract_version: "launch-monitor-player-covariation/1.0.0",
        analysis_kind: "selected_pair",
        status: "available",
        request: {},
        pooled: {},
        within_player: {},
        between_player: {},
        per_player: [],
        meta_analysis: {},
        missingness: {},
        units: {},
        lineage: { backing_records: rows.map((_, index) => ({ row_index: index })) },
        availability: [],
        uncertainty: {},
        player_identity: { trust_level: "explicit_user_attested" },
        vendor_provenance: {},
        claims: { device_emulation: false, device_certification: false, causal_inference: false },
        definitions: {},
        warnings: [],
      }),
    })));
    render(<LaunchMonitorPlayerWorkspace rows={rows} sourceName="test.csv" />);
    fireEvent.change(screen.getByLabelText("Canonical Upstream authority URL"), { target: { value: "https://upstream.example" } });
    fireEvent.change(screen.getByLabelText("Player identity column"), { target: { value: "player_id" } });
    fireEvent.click(screen.getByLabelText(/I attest/i));
    fireEvent.click(screen.getByRole("button", { name: /run canonical player covariation/i }));
    expect(await screen.findByLabelText("Canonical player covariation evidence")).toHaveTextContent("explicit_user_attested");
    expect(screen.getByRole("button", { name: /export full bundle/i })).toBeEnabled();
  });
});
