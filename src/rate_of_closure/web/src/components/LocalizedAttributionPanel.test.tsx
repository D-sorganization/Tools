import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import fixture from "../model/__fixtures__/localized_attribution_authority_v1.json";
import { attributionAuthorityFromValue } from "../model/localizedAttribution";
import { LocalizedAttributionPanel } from "./LocalizedAttributionPanel";

describe("LocalizedAttributionPanel", () => {
  it("fails closed when a localized Monte Carlo run lacks isolated pairs", () => {
    render(<LocalizedAttributionPanel authority={null} localizedRunAvailable />);

    expect(screen.getByRole("status")).toHaveTextContent(/attribution unavailable/i);
    expect(screen.getByText(/scatter and rank correlation are not substituted/i))
      .toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("selects source, state/impact/shot target, and retained pair accessibly", async () => {
    const user = userEvent.setup();
    render(<LocalizedAttributionPanel
      authority={attributionAuthorityFromValue(fixture)}
      localizedRunAvailable
    />);

    expect(screen.getByText(/paired planted-intervention response only/i))
      .toBeInTheDocument();
    expect(screen.getByText(/joint.shoulder · \[0.001, 0.003\) s/i))
      .toBeInTheDocument();
    expect(screen.getByText(/swing.clubhead.reference at 0.002 s/i))
      .toBeInTheDocument();
    expect(screen.getByText(/2\/3 available/i)).toBeInTheDocument();

    await user.selectOptions(
      screen.getByLabelText("Localized attribution target"),
      "impact.clubhead_speed",
    );
    await user.selectOptions(
      screen.getByLabelText("Localized attribution retained pair"),
      "0:2",
    );

    expect(screen.getByText("no_impact_unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(/Unavailable/)).toHaveLength(2);
    expect(screen.getByText(/1\/3 available/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /export raw observations csv/i }))
      .toBeEnabled();
    expect(screen.getByRole("button", { name: /export view definition json/i }))
      .toBeEnabled();
  });
});
