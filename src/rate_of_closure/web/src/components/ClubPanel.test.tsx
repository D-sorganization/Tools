import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { getClub, type ClubSpec } from "../model/club";
import { ClubPanel } from "./ClubPanel";

describe("ClubPanel canonical state", () => {
  it("restores the selected club after its primary tab unmounts", async () => {
    let current: ClubSpec = getClub("Driver 10.5°");
    const onSpecChange = vi.fn((spec: ClubSpec) => { current = spec; });
    const first = render(<ClubPanel initialSpec={current}
      onDriveScenario={() => undefined} onGenerate={() => undefined}
      onSpecChange={onSpecChange} />);

    fireEvent.change(screen.getByRole("combobox", { name: "Club" }), {
      target: { value: "Pitching Wedge" },
    });
    await waitFor(() => expect(current.name).toBe("Pitching Wedge"));
    first.unmount();

    render(<ClubPanel initialSpec={current}
      onDriveScenario={() => undefined} onGenerate={() => undefined} />);
    expect(screen.getByRole("combobox", { name: "Club" }))
      .toHaveValue("Pitching Wedge");
  });
});
