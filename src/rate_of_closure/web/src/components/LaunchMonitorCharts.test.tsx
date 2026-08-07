import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { sessionTrend } from "../model/launchMonitorPlayerAnalytics";
import { SessionTrendPlot } from "./LaunchMonitorCharts";

describe("SessionTrendPlot", () => {
  it("renders a separately labelled series for each player", () => {
    const trend = sessionTrend([
      { player_id: "alpha", session_id: "1", ball_speed: 100 },
      { player_id: "alpha", session_id: "2", ball_speed: 110 },
      { player_id: "beta", session_id: "1", ball_speed: 200 },
      { player_id: "beta", session_id: "2", ball_speed: 190 },
    ], "ball_speed")!;
    render(<SessionTrendPlot trend={trend} />);
    expect(screen.getByLabelText("Player alpha")).toBeInTheDocument();
    expect(screen.getByLabelText("Player beta")).toBeInTheDocument();
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
  });
});

