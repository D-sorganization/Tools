import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterlocksPanel } from "./InterlocksPanel";
import type { InterlockConfig } from "../App";

const interlocks: InterlockConfig[] = Array.from({ length: 32 }, () => ({
  lolo_limit: null,
  low_limit: null,
  high_limit: null,
  hihi_limit: null,
}));

function renderPanel(props: { tripped?: boolean; firstOutTagId?: number | null }) {
  return render(
    <InterlocksPanel
      interlocks={interlocks}
      onChange={vi.fn()}
      onDeploy={vi.fn()}
      deploying={false}
      {...props}
    />,
  );
}

describe("InterlocksPanel first-out indicator (F07)", () => {
  it("shows the first-out tag name when tripped and the tag is known", () => {
    renderPanel({ tripped: true, firstOutTagId: 7 });
    expect(screen.getByTestId("first-out")).toHaveTextContent("Tag 7");
  });

  it("shows tag 0 (falsy id) rather than treating it as unknown", () => {
    renderPanel({ tripped: true, firstOutTagId: 0 });
    expect(screen.getByTestId("first-out")).toHaveTextContent("Tag 0");
  });

  it("shows an explicit unknown state when tripped but the read failed", () => {
    renderPanel({ tripped: true, firstOutTagId: null });
    expect(screen.getByTestId("first-out")).toHaveTextContent(/unknown/i);
  });

  it("shows unknown when tripped and the field is absent", () => {
    renderPanel({ tripped: true });
    expect(screen.getByTestId("first-out")).toHaveTextContent(/unknown/i);
  });

  it("shows no first-out when not tripped", () => {
    renderPanel({ tripped: false, firstOutTagId: null });
    expect(screen.queryByTestId("first-out")).toBeNull();
  });
});
