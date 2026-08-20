import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { LaunchMonitorPlayerWorkspace } from "./LaunchMonitorPlayerWorkspace";

const rows: LaunchMonitorRow[] = Array.from({ length: 12 }, (_, index) => ({
  shot_id: `s${index}`,
  player_id: index < 6 ? "p1" : "p2",
  face_angle: index,
  club_path: index * 0.8 + (index % 2),
}));

describe("LaunchMonitorPlayerWorkspace", () => {
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
});
