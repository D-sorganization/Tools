import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LaunchMonitorCovariation } from "./LaunchMonitorCovariation";

const rows = [
  ...[0, 1, 2, 3].map((value) => ({ player_id: "alpha", club_path: value, face_angle: value + 1, ball_speed: 100 + value })),
  ...[4, 5, 6, 7].map((value) => ({ player_id: "beta", club_path: value, face_angle: value + 1, ball_speed: 100 + value })),
];

describe("LaunchMonitorCovariation", () => {
  it("offers accessible arbitrary-variable and statistical controls", () => {
    render(<LaunchMonitorCovariation rows={rows} />);
    expect(screen.getByLabelText("Covariation player column")).toHaveValue("player_id");
    expect(screen.getByLabelText("Covariation X variable")).toHaveValue("club_path");
    expect(screen.getByLabelText("Covariation Y variable")).toHaveValue("face_angle");
    expect(screen.getByLabelText("Covariation coefficient method")).toHaveValue("pearson");
    expect(screen.getByLabelText("Covariation minimum player sample count")).toHaveAttribute("title");
    expect(screen.getByLabelText("Covariation confidence level")).toHaveAttribute("title");
  });

  it("shows within-player, population, meta, visualization, and backing exports", () => {
    render(<LaunchMonitorCovariation rows={rows} />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze Player Covariation" }));
    expect(screen.getByText("Within-player centered")).toBeInTheDocument();
    expect(screen.getByText("Pooled raw")).toBeInTheDocument();
    expect(screen.getByText("Between-player means")).toBeInTheDocument();
    expect(screen.getByText("Random-effects meta-analysis")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /within-player.*scatter/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export Covariation CSV" })).toHaveAttribute("title");
    expect(screen.getByRole("button", { name: "Export Covariation JSON" })).toHaveAttribute("title");
    expect(screen.getByText(/does not establish causality/i)).toBeInTheDocument();
  });

  it("scans and ranks every selected numeric pair with a multiplicity warning", () => {
    render(<LaunchMonitorCovariation rows={rows} />);
    fireEvent.click(screen.getByRole("button", { name: "Rank Variable Pairs" }));
    expect(screen.getByText("Ranked Population Pairs")).toBeInTheDocument();
    expect(screen.getByText(/exploratory.*multiple comparisons/i)).toBeInTheDocument();
  });
});
