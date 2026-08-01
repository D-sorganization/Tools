import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DataAgeIndicator } from "./DataAgeIndicator";

/**
 * The header liveness pill (#4010).
 *
 * The old pill was a boolean: CONNECTED or OFFLINE. A dead backend serving its
 * never-cleared `{}` kept it on CONNECTED, so a frozen process value read as a
 * beautifully stable one. The pill now always shows the AGE of the data, and a
 * frozen stream gets its own third state that neither of the old two could
 * express.
 */

describe("DataAgeIndicator", () => {
  it("shows CONNECTED with the age while data is fresh", () => {
    render(<DataAgeIndicator freshness="live" ageMs={1200} />);
    expect(screen.getByText("CONNECTED")).toBeInTheDocument();
    expect(screen.getByText(/1 s ago/)).toBeInTheDocument();
  });

  it("shows STALE DATA with the age once the stream freezes", () => {
    render(<DataAgeIndicator freshness="stale" ageMs={20 * 60_000 + 4_000} />);
    expect(screen.getByText("STALE DATA")).toBeInTheDocument();
    // The operator must be able to read HOW stale — twenty minutes, not "a bit".
    expect(screen.getByText(/20 m 4 s ago/)).toBeInTheDocument();
    expect(screen.queryByText("CONNECTED")).not.toBeInTheDocument();
  });

  it("shows OFFLINE when no frame has ever arrived", () => {
    render(<DataAgeIndicator freshness="offline" ageMs={undefined} />);
    expect(screen.getByText("OFFLINE")).toBeInTheDocument();
    expect(screen.getByText(/no data/i)).toBeInTheDocument();
  });

  it("exposes the state to assistive tech and to CSS", () => {
    const { container } = render(<DataAgeIndicator freshness="stale" ageMs={9000} />);
    const root = container.querySelector("[data-freshness]");
    expect(root).toHaveAttribute("data-freshness", "stale");
    expect(root).toHaveAttribute("role", "status");
  });
});
